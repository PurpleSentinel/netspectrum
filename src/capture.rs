//! Passive capture thread. Opens the interface via libpcap (never transmits),
//! parses L2-L4 headers only (snaplen 96 bytes -- payloads are never read),
//! and streams compact PacketMeta records to the render thread.

use std::collections::HashSet;
use std::net::IpAddr;
use std::process::exit;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use etherparse::{InternetSlice, LinkSlice, SlicedPacket, TransportSlice};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum L4 {
    Tcp,
    Udp,
    Icmp,
    Other,
}

#[derive(Copy, Clone, Debug)]
pub struct PacketMeta {
    /// Original wire length in bytes (not the truncated snaplen capture).
    pub len: u32,
    pub src: IpAddr,
    pub dst: IpAddr,
    pub sport: u16,
    pub dport: u16,
    pub l4: L4,
    pub ethertype: u16,
    pub inbound: bool,
}

const UNSPEC: IpAddr = IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED);

/// Parse one packet into the minimal metadata the visualiser needs.
///
/// The capture snaplen is small, so this deliberately stays at Ethernet/IP/L4
/// headers and never depends on payload bytes.
fn parse(data: &[u8], wire_len: u32, local: &HashSet<IpAddr>) -> Option<PacketMeta> {
    let sliced = SlicedPacket::from_ethernet(data).ok()?;

    let ethertype = match &sliced.link {
        Some(LinkSlice::Ethernet2(e)) => e.ether_type(),
        _ => 0,
    };

    let (src, dst) = match &sliced.ip {
        Some(InternetSlice::Ipv4(h, _)) => (
            IpAddr::V4(h.source_addr()),
            IpAddr::V4(h.destination_addr()),
        ),
        Some(InternetSlice::Ipv6(h, _)) => (
            IpAddr::V6(h.source_addr()),
            IpAddr::V6(h.destination_addr()),
        ),
        None => (UNSPEC, UNSPEC),
    };

    let (l4, sport, dport) = match &sliced.transport {
        Some(TransportSlice::Tcp(t)) => (L4::Tcp, t.source_port(), t.destination_port()),
        Some(TransportSlice::Udp(u)) => (L4::Udp, u.source_port(), u.destination_port()),
        Some(TransportSlice::Icmpv4(_)) | Some(TransportSlice::Icmpv6(_)) => (L4::Icmp, 0, 0),
        _ => (L4::Other, 0, 0),
    };

    // Direction: if the source is one of our addresses it's outbound,
    // otherwise (including broadcast/multicast) treat as inbound.
    let inbound = !local.contains(&src);

    Some(PacketMeta {
        len: wire_len,
        src,
        dst,
        sport,
        dport,
        l4,
        ethertype,
        inbound,
    })
}

/// Start the passive pcap loop and return the receiving end of the packet
/// metadata channel used by the render thread.
pub fn spawn(
    iface: String,
    filter: Option<String>,
    local: HashSet<IpAddr>,
) -> Receiver<PacketMeta> {
    // Bound the queue so a stalled UI cannot grow memory without limit. Capture
    // uses try_send below and drops frames instead of blocking.
    let (tx, rx): (Sender<PacketMeta>, Receiver<PacketMeta>) = bounded(1 << 16);

    std::thread::Builder::new()
        .name("capture".into())
        .spawn(move || {
            let cap = pcap::Capture::from_device(iface.as_str()).and_then(|c| {
                c.promisc(true)
                    .immediate_mode(true)
                    .snaplen(96) // headers only; payloads never captured
                    .open()
            });

            // Capture setup errors are fatal: without a pcap handle the UI would
            // just render an idle display forever.
            let mut cap = match cap {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: cannot open '{iface}' for capture: {e}");
                    eprintln!("hint:  run with sudo, or grant the binary capture rights:");
                    eprintln!("       sudo setcap cap_net_raw,cap_net_admin+ep ./netspectrum");
                    exit(1);
                }
            };

            if let Some(f) = filter {
                // Let libpcap compile the user-provided BPF program so invalid
                // syntax fails early and loudly.
                if let Err(e) = cap.filter(&f, true) {
                    eprintln!("error: bad BPF filter '{f}': {e}");
                    exit(1);
                }
            }

            loop {
                match cap.next_packet() {
                    Ok(pkt) => {
                        if let Some(meta) = parse(pkt.data, pkt.header.len, &local) {
                            // If the UI can't keep up, drop rather than block capture.
                            if let Err(TrySendError::Disconnected(_)) = tx.try_send(meta) {
                                return;
                            }
                        }
                    }
                    Err(pcap::Error::TimeoutExpired) => continue,
                    Err(e) => {
                        eprintln!("capture error: {e}");
                        return;
                    }
                }
            }
        })
        .expect("spawn capture thread");

    rx
}
