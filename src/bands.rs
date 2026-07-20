//! The signal chain: classify packets into bands per mode, integrate bytes,
//! log-scale against a slowly-adapting reference level (auto-gain), then run
//! audio-analyser ballistics (fast attack / slow release) plus peak-hold caps.

use std::collections::HashMap;
use std::net::IpAddr;

use crate::capture::{PacketMeta, L4};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Protocol,
    Ports,
    Hosts,
    Hybrid,
    Sizes,
}

impl Mode {
    /// Uppercase label shown in the HUD for the current traffic grouping.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Protocol => "PROTOCOLS",
            Mode::Ports => "PORT SPECTRUM",
            Mode::Hosts => "TOP HOSTS",
            Mode::Hybrid => "PROTOCOL x DIRECTION",
            Mode::Sizes => "PACKET SIZES",
        }
    }

    /// Parse the CLI-friendly mode aliases accepted by `-m/--mode`.
    pub fn from_str(s: &str) -> Option<Mode> {
        match s.to_ascii_lowercase().as_str() {
            "protocol" | "protocols" | "proto" => Some(Mode::Protocol),
            "ports" | "port" => Some(Mode::Ports),
            "hosts" | "host" => Some(Mode::Hosts),
            "hybrid" | "direction" => Some(Mode::Hybrid),
            "sizes" | "size" => Some(Mode::Sizes),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- protocol map

const PROTO_LABELS: [&str; 18] = [
    "ARP",
    "ICMP",
    "DHCP",
    "NTP",
    "DNS",
    "mDNS/SSDP",
    "HTTP",
    "TLS",
    "QUIC",
    "SSH",
    "RDP/VNC",
    "SMB/NFS",
    "MAIL",
    "DB",
    "SYSLOG",
    "TCP other",
    "UDP other",
    "other",
];
const ARP: usize = 0;
const ICMP: usize = 1;
const DHCP: usize = 2;
const NTP: usize = 3;
const DNS: usize = 4;
const DISC: usize = 5;
const HTTP: usize = 6;
const TLS: usize = 7;
const QUIC: usize = 8;
const SSH: usize = 9;
const REMOTE: usize = 10;
const FILE: usize = 11;
const MAIL: usize = 12;
const DB: usize = 13;
const SYSLOG: usize = 14;
const OTHER_TCP: usize = 15;
const OTHER_UDP: usize = 16;
const OTHER: usize = 17;

/// Map common L4 service ports and ARP/ICMP into the fixed protocol bands.
fn proto_band(m: &PacketMeta) -> usize {
    if m.ethertype == 0x0806 {
        return ARP;
    }
    match m.l4 {
        L4::Icmp => ICMP,
        L4::Other => OTHER,
        L4::Tcp | L4::Udp => {
            let p = |x: u16| m.sport == x || m.dport == x;
            if p(53) {
                DNS
            } else if p(5353) || p(1900) || p(5355) {
                DISC
            } else if p(67) || p(68) || p(546) || p(547) {
                DHCP
            } else if p(123) {
                NTP
            } else if m.l4 == L4::Udp && p(443) {
                QUIC
            } else if p(443) || p(8443) {
                TLS
            } else if p(80) || p(8080) || p(8000) {
                HTTP
            } else if p(22) {
                SSH
            } else if p(3389) || p(5900) {
                REMOTE
            } else if p(445) || p(139) || p(2049) {
                FILE
            } else if p(25) || p(465) || p(587) || p(143) || p(993) || p(110) || p(995) {
                MAIL
            } else if p(5432) || p(3306) || p(6379) || p(27017) || p(9200) {
                DB
            } else if p(514) || p(6514) {
                SYSLOG
            } else if m.l4 == L4::Tcp {
                OTHER_TCP
            } else {
                OTHER_UDP
            }
        }
    }
}

// Condensed groups for the mirrored hybrid view.
const GROUP_LABELS: [&str; 8] = [
    "WEB",
    "DNS",
    "REMOTE",
    "MAIL",
    "INFRA",
    "DISCOVERY",
    "DATA",
    "OTHER",
];

/// Collapse the detailed protocol map into the smaller directional hybrid set.
fn group_band(m: &PacketMeta) -> usize {
    match proto_band(m) {
        HTTP | TLS | QUIC => 0,
        DNS => 1,
        SSH | REMOTE => 2,
        MAIL => 3,
        ARP | ICMP | DHCP | NTP | SYSLOG => 4,
        DISC => 5,
        FILE | DB => 6,
        _ => 7,
    }
}

const PORT_LABELS: [&str; 16] = [
    "0-1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1k",
    "1k-2k", "2k-4k", "4k-8k", "8k-16k", "16k-32k", "32k+",
];

/// Bucket traffic by the lower endpoint port using log2-style ranges.
fn port_band(m: &PacketMeta) -> usize {
    let sp = if m.sport == 0 {
        m.dport
    } else if m.dport == 0 {
        m.sport
    } else {
        m.sport.min(m.dport)
    };
    if sp == 0 {
        0
    } else {
        (15 - (sp.leading_zeros() as usize)).min(15)
    }
}

const SIZE_EDGES: [u32; 7] = [64, 128, 256, 512, 1024, 1280, 1518];
const SIZE_LABELS: [&str; 8] = [
    "<=64",
    "65-128",
    "129-256",
    "257-512",
    "513-1k",
    "1k-1.2k",
    "1.2k-1.5k",
    "jumbo",
];

/// Convert captured wire length into packet-size histogram bands.
fn size_band(m: &PacketMeta) -> usize {
    for (i, e) in SIZE_EDGES.iter().enumerate() {
        if m.len <= *e {
            return i;
        }
    }
    SIZE_EDGES.len()
}

// ---------------------------------------------------------------- ballistics

const ATTACK_TAU: f32 = 0.035; // seconds -- near-instant rise
const RELEASE_TAU: f32 = 0.30; // seconds -- smooth fall
const PEAK_HOLD: f32 = 1.1; // seconds the peak cap hangs
const PEAK_FALL: f32 = 0.55; // full-scale units per second afterwards
const GAIN_TAU: f32 = 12.0; // seconds -- reference level decay
const RATE_FLOOR: f64 = 25_000.0; // bytes/sec -- min full-scale so idle links stay calm

pub struct Spectrum {
    pub mode: Mode,
    pub labels: Vec<String>,
    pub mirrored: bool,
    /// Display values 0..1 per bar (for hybrid: [up0, down0, up1, down1, ...]).
    pub disp: Vec<f32>,
    pub peak: Vec<f32>,
    peak_hold_until: Vec<f32>,
    acc: Vec<f64>,      // bytes accumulated since last tick
    rate_ema: Vec<f64>, // smoothed bytes/sec for readouts
    max_rate: f64,      // auto-gain reference (full scale)
    clock: f32,

    pub rate_in: f64,
    pub rate_out: f64,
    acc_in: f64,
    acc_out: f64,

    n_hosts: usize,
    host_bytes: HashMap<IpAddr, f64>,
    host_slots: Vec<Option<IpAddr>>,
    host_refresh: f32,
    pub labels_dirty: bool,
}

impl Spectrum {
    /// Create a signal processor with the requested initial mode.
    pub fn new(mode: Mode, n_hosts: usize) -> Self {
        let mut s = Spectrum {
            mode,
            labels: vec![],
            mirrored: false,
            disp: vec![],
            peak: vec![],
            peak_hold_until: vec![],
            acc: vec![],
            rate_ema: vec![],
            max_rate: RATE_FLOOR,
            clock: 0.0,
            rate_in: 0.0,
            rate_out: 0.0,
            acc_in: 0.0,
            acc_out: 0.0,
            n_hosts: n_hosts.clamp(4, 24),
            host_bytes: HashMap::new(),
            host_slots: vec![],
            host_refresh: 0.0,
            labels_dirty: true,
        };
        s.set_mode(mode);
        s
    }

    /// Switch band layout and reset display state so old-mode energy does not
    /// leak into the new view.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.mirrored = mode == Mode::Hybrid;
        self.labels = match mode {
            Mode::Protocol => PROTO_LABELS.iter().map(|s| s.to_string()).collect(),
            Mode::Ports => PORT_LABELS.iter().map(|s| s.to_string()).collect(),
            Mode::Sizes => SIZE_LABELS.iter().map(|s| s.to_string()).collect(),
            Mode::Hybrid => GROUP_LABELS.iter().map(|s| s.to_string()).collect(),
            Mode::Hosts => {
                self.host_slots = vec![None; self.n_hosts];
                (0..self.n_hosts).map(|_| "-".to_string()).collect()
            }
        };
        let bars = self.bar_count();
        self.disp = vec![0.0; bars];
        self.peak = vec![0.0; bars];
        self.peak_hold_until = vec![0.0; bars];
        self.acc = vec![0.0; bars];
        self.rate_ema = vec![0.0; bars];
        self.max_rate = RATE_FLOOR;
        self.labels_dirty = true;
    }

    /// Number of drawable bars (hybrid draws two per labelled band).
    pub fn bar_count(&self) -> usize {
        if self.mirrored {
            self.labels.len() * 2
        } else {
            self.labels.len()
        }
    }

    pub fn band_count(&self) -> usize {
        self.labels.len()
    }

    /// Add one packet's wire length into the accumulator for the active mode.
    pub fn ingest(&mut self, m: &PacketMeta) {
        let bytes = m.len as f64;
        if m.inbound {
            self.acc_in += bytes;
        } else {
            self.acc_out += bytes;
        }

        match self.mode {
            Mode::Protocol => self.acc[proto_band(m)] += bytes,
            Mode::Ports => self.acc[port_band(m)] += bytes,
            Mode::Sizes => self.acc[size_band(m)] += bytes,
            Mode::Hybrid => {
                let g = group_band(m);
                let idx = g * 2 + if m.inbound { 0 } else { 1 };
                self.acc[idx] += bytes;
            }
            Mode::Hosts => {
                // Hosts mode ranks the remote endpoint, not the local address.
                let remote = if m.inbound { m.src } else { m.dst };
                *self.host_bytes.entry(remote).or_insert(0.0) += bytes;
                if let Some(slot) = self.host_slots.iter().position(|s| *s == Some(remote)) {
                    self.acc[slot] += bytes;
                }
            }
        }
    }

    /// Advance the signal chain: convert accumulated bytes into smoothed,
    /// log-scaled display values and peak caps.
    pub fn tick(&mut self, dt: f32) {
        let dt = dt.clamp(0.0005, 0.25);
        self.clock += dt;

        // Totals (smoothed over ~0.5 s).
        let k_tot = 1.0 - (-dt / 0.5).exp();
        self.rate_in += (self.acc_in / dt as f64 - self.rate_in) * k_tot as f64;
        self.rate_out += (self.acc_out / dt as f64 - self.rate_out) * k_tot as f64;
        self.acc_in = 0.0;
        self.acc_out = 0.0;

        // Auto-gain: reference decays slowly, is pushed up instantly by louder bands.
        self.max_rate = (self.max_rate * (-dt as f64 / GAIN_TAU as f64).exp()).max(RATE_FLOOR);

        let k_attack = 1.0 - (-dt / ATTACK_TAU).exp();
        let k_release = 1.0 - (-dt / RELEASE_TAU).exp();
        let k_rate = 1.0 - (-dt / 0.5).exp();

        let rates: Vec<f64> = self.acc.iter().map(|b| b / dt as f64).collect();
        for r in &rates {
            if *r > self.max_rate {
                self.max_rate = *r;
            }
        }
        let log_full = (1.0 + self.max_rate).ln();

        for (i, rate) in rates.iter().enumerate() {
            self.rate_ema[i] += (*rate - self.rate_ema[i]) * k_rate as f64;

            // Log scaling keeps quiet and busy links both visually active.
            let norm = (((1.0 + *rate).ln() / log_full) as f32).clamp(0.0, 1.0);
            let k = if norm > self.disp[i] {
                k_attack
            } else {
                k_release
            };
            self.disp[i] += (norm - self.disp[i]) * k;

            if self.disp[i] >= self.peak[i] {
                self.peak[i] = self.disp[i];
                self.peak_hold_until[i] = self.clock + PEAK_HOLD;
            } else if self.clock > self.peak_hold_until[i] {
                self.peak[i] = (self.peak[i] - PEAK_FALL * dt).max(0.0);
            }
            self.acc[i] = 0.0;
        }

        if self.mode == Mode::Hosts {
            self.host_tick(dt);
        }
    }

    /// Periodically refresh host slots while preserving stable bar positions.
    fn host_tick(&mut self, dt: f32) {
        // Decay ranking scores so stale hosts fade out (~30 s half-life-ish).
        let decay = (-dt as f64 / 30.0).exp();
        self.host_bytes.retain(|_, v| {
            *v *= decay;
            *v > 512.0
        });

        self.host_refresh -= dt;
        if self.host_refresh > 0.0 {
            return;
        }
        self.host_refresh = 0.5;

        let mut ranked: Vec<(IpAddr, f64)> =
            self.host_bytes.iter().map(|(k, v)| (*k, *v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<IpAddr> = ranked
            .iter()
            .take(self.n_hosts)
            .map(|(ip, _)| *ip)
            .collect();

        // Hysteresis: keep an occupant if it's still in the top 2N; fill gaps with
        // the best newcomers so bars don't reshuffle constantly.
        let keep: Vec<IpAddr> = ranked
            .iter()
            .take(self.n_hosts * 2)
            .map(|(ip, _)| *ip)
            .collect();
        let mut changed = false;
        for slot in self.host_slots.iter_mut() {
            if let Some(ip) = slot {
                if !keep.contains(ip) {
                    *slot = None;
                    changed = true;
                }
            }
        }
        for ip in top {
            if self.host_slots.contains(&Some(ip)) {
                continue;
            }
            if let Some(free) = self.host_slots.iter().position(|s| s.is_none()) {
                self.host_slots[free] = Some(ip);
                changed = true;
            }
        }
        if changed {
            for (i, slot) in self.host_slots.iter().enumerate() {
                self.labels[i] = match slot {
                    Some(ip) => shorten_ip(ip),
                    None => "-".to_string(),
                };
                self.disp[i] = 0.0;
                self.peak[i] = 0.0;
                self.rate_ema[i] = 0.0;
            }
            self.labels_dirty = true;
        }
    }

    /// Smoothed per-band rate for the readout under each label.
    pub fn band_rate(&self, band: usize) -> f64 {
        if self.mirrored {
            self.rate_ema[band * 2] + self.rate_ema[band * 2 + 1]
        } else {
            self.rate_ema[band]
        }
    }
}

/// Shorten long IPv6 addresses so band labels remain readable.
fn shorten_ip(ip: &IpAddr) -> String {
    let s = ip.to_string();
    if s.len() > 16 {
        format!("{}\u{2026}", &s[..15])
    } else {
        s
    }
}

/// Format bytes/sec for the HUD and per-band labels.
pub fn human_rate(bps: f64) -> String {
    if bps >= 1e9 {
        format!("{:.2} GB/s", bps / 1e9)
    } else if bps >= 1e6 {
        format!("{:.1} MB/s", bps / 1e6)
    } else if bps >= 1e3 {
        format!("{:.1} KB/s", bps / 1e3)
    } else {
        format!("{:.0} B/s", bps)
    }
}
