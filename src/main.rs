mod audio;
mod bands;
mod capture;
mod render;

use std::collections::HashSet;
use std::net::IpAddr;

use anyhow::{bail, Context, Result};
use clap::Parser;

use audio::AudioConfig;
use bands::{Mode, Spectrum};

/// A GPU-accelerated graphic equaliser for passively observed network traffic.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Interface to capture on (defaults to the first usable device)
    #[arg(short, long)]
    interface: Option<String>,

    /// Initial view: protocol | ports | hosts | hybrid | sizes
    #[arg(short, long, default_value = "protocol")]
    mode: String,

    /// Optional BPF filter, e.g. "not port 22" to hide your own SSH session
    #[arg(short, long)]
    filter: Option<String>,

    /// Number of bands in hosts mode
    #[arg(long, default_value_t = 12)]
    hosts: usize,

    /// List capture interfaces and exit
    #[arg(long)]
    list: bool,

    /// Optional audio output target for pw-cat/pacat, e.g. an HDMI sink name
    #[arg(long)]
    audio_output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let devices = pcap::Device::list().context("listing capture devices")?;

    if args.list {
        for d in &devices {
            let addrs: Vec<String> = d.addresses.iter().map(|a| a.addr.to_string()).collect();
            println!("{:16} {}", d.name, addrs.join(", "));
        }
        return Ok(());
    }

    let mode = Mode::from_str(&args.mode).with_context(|| {
        format!(
            "unknown mode '{}' (protocol|ports|hosts|hybrid|sizes)",
            args.mode
        )
    })?;

    let device = match &args.interface {
        Some(name) => devices
            .iter()
            .find(|d| &d.name == name)
            .cloned()
            .with_context(|| format!("interface '{name}' not found (try --list)"))?,
        None => pcap::Device::lookup()
            .context("looking up default device")?
            .context("no capture device available (try --list, run as root)")?,
    };

    if device.addresses.is_empty() {
        eprintln!(
            "note: '{}' reports no addresses; direction (in/out) detection will treat all traffic as inbound",
            device.name
        );
    }

    let local: HashSet<IpAddr> = device.addresses.iter().map(|a| a.addr).collect();
    let iface = device.name.clone();

    if args.hosts < 4 || args.hosts > 24 {
        bail!("--hosts must be between 4 and 24");
    }

    println!("netspectrum: passive capture on '{iface}' -- no packets are ever transmitted");

    let rx = capture::spawn(iface.clone(), args.filter.clone(), local);
    let spectrum = Spectrum::new(mode, args.hosts);
    render::run(
        spectrum,
        rx,
        iface,
        AudioConfig {
            output: args.audio_output,
        },
    )
}
