use std::f32::consts::TAU;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::JoinHandle;

use anyhow::{Context, Result};

use crate::bands::Spectrum;

const SAMPLE_RATE: f32 = 48_000.0;
const CHANNELS: usize = 2;
const FRAMES_PER_BLOCK: usize = 256;

#[derive(Copy, Clone, Debug, Default)]
struct AudioTarget {
    enabled: bool,
    mono_level: f32,
    inbound_level: f32,
    outbound_level: f32,
    centroid: f32,
}

#[derive(Copy, Clone, Debug)]
struct AudioEngineState {
    target: AudioTarget,
    amp: f32,
    mono_level: f32,
    inbound_level: f32,
    outbound_level: f32,
    centroid: f32,
    mono_phase: f32,
    inbound_phase: f32,
    outbound_phase: f32,
}

impl Default for AudioEngineState {
    fn default() -> Self {
        Self {
            target: AudioTarget::default(),
            amp: 0.0,
            mono_level: 0.0,
            inbound_level: 0.0,
            outbound_level: 0.0,
            centroid: 0.5,
            mono_phase: 0.0,
            inbound_phase: 0.0,
            outbound_phase: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AudioSnapshot {
    pub enabled: bool,
    pub available: bool,
}

pub struct AudioState {
    shared: Arc<Mutex<AudioEngineState>>,
    running: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    child: Child,
}

impl AudioState {
    pub fn new() -> Result<Self> {
        let mut child = spawn_audio_helper().context("starting audio playback helper")?;

        let stdin = child.stdin.take().context("opening audio helper stdin")?;
        let shared = Arc::new(Mutex::new(AudioEngineState::default()));
        let running = Arc::new(AtomicBool::new(true));
        let healthy = Arc::new(AtomicBool::new(true));
        let worker = {
            let shared = shared.clone();
            let running = running.clone();
            let healthy = healthy.clone();
            std::thread::Builder::new()
                .name("audio".into())
                .spawn(move || audio_worker(stdin, shared, running, healthy))
                .context("spawning audio worker")?
        };

        Ok(Self {
            shared,
            running,
            healthy,
            worker: Some(worker),
            child,
        })
    }

    pub fn update(&mut self, spectrum: &Spectrum) {
        self.set_target(target_from_spectrum(spectrum, true));
    }

    pub fn snapshot(&self) -> AudioSnapshot {
        AudioSnapshot {
            enabled: self.healthy.load(Ordering::Relaxed),
            available: self.healthy.load(Ordering::Relaxed),
        }
    }

    fn set_target(&self, target: AudioTarget) {
        if let Ok(mut state) = self.shared.lock() {
            state.target = target;
        }
    }
}

impl Drop for AudioState {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.child.kill();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub enum AudioControl {
    Off,
    On(AudioState),
    Unavailable,
}

impl AudioControl {
    pub fn new() -> Self {
        Self::Off
    }

    pub fn toggle(&mut self) -> AudioSnapshot {
        *self = match std::mem::replace(self, Self::Off) {
            Self::Off | Self::Unavailable => match AudioState::new() {
                Ok(audio) => Self::On(audio),
                Err(e) => {
                    eprintln!("audio unavailable: {e}");
                    Self::Unavailable
                }
            },
            Self::On(_) => Self::Off,
        };
        self.snapshot()
    }

    pub fn update(&mut self, spectrum: &Spectrum) {
        if let Self::On(audio) = self {
            audio.update(spectrum);
            if !audio.snapshot().available {
                *self = Self::Unavailable;
            }
        }
    }

    pub fn snapshot(&self) -> AudioSnapshot {
        match self {
            Self::Off => AudioSnapshot {
                enabled: false,
                available: true,
            },
            Self::On(audio) => audio.snapshot(),
            Self::Unavailable => AudioSnapshot {
                enabled: false,
                available: false,
            },
        }
    }
}

fn audio_worker(
    mut output: impl Write,
    shared: Arc<Mutex<AudioEngineState>>,
    running: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
) {
    let mut block = vec![0u8; FRAMES_PER_BLOCK * CHANNELS * std::mem::size_of::<f32>()];

    while running.load(Ordering::Relaxed) {
        let Ok(mut state) = shared.lock() else {
            fill_silence(&mut block);
            if output.write_all(&block).is_err() {
                healthy.store(false, Ordering::Relaxed);
                return;
            }
            continue;
        };

        fill_audio_block(&mut block, &mut state);
        drop(state);

        if output.write_all(&block).is_err() {
            healthy.store(false, Ordering::Relaxed);
            return;
        }
    }
}

fn spawn_audio_helper() -> Result<Child> {
    let attempts = [
        (
            "pw-cat",
            &[
                "--raw",
                "--playback",
                "--rate",
                "48000",
                "--channels",
                "2",
                "--format",
                "f32",
                "--latency",
                "20ms",
                "-",
            ][..],
        ),
        (
            "pacat",
            &[
                "--raw",
                "--playback",
                "--rate",
                "48000",
                "--channels",
                "2",
                "--format",
                "float32le",
                "--client-name",
                "netspectrum",
                "--stream-name",
                "netspectrum audio",
            ][..],
        ),
    ];

    let mut errors = Vec::new();
    for (program, args) in attempts {
        match Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(child) => return Ok(child),
            Err(e) => errors.push(format!("{program}: {e}")),
        }
    }

    anyhow::bail!("no supported audio helper found ({})", errors.join("; "))
}

fn fill_audio_block(block: &mut [u8], state: &mut AudioEngineState) {
    for frame in block.chunks_exact_mut(CHANNELS * std::mem::size_of::<f32>()) {
        let target = state.target;
        let gate = if target.enabled { 1.0 } else { 0.0 };
        state.amp += (gate - state.amp) * 0.004;
        state.mono_level += (target.mono_level - state.mono_level) * 0.0025;
        state.inbound_level += (target.inbound_level - state.inbound_level) * 0.0025;
        state.outbound_level += (target.outbound_level - state.outbound_level) * 0.0025;
        state.centroid += (target.centroid - state.centroid) * 0.0015;

        let mono = soft_tone(
            &mut state.mono_phase,
            130.0 + state.centroid * 520.0 + state.mono_level * 120.0,
        ) * state.mono_level
            * 0.11;

        let inbound = soft_tone(
            &mut state.inbound_phase,
            330.0 + state.inbound_level * 420.0,
        ) * state.inbound_level
            * 0.09;

        let outbound = soft_tone(
            &mut state.outbound_phase,
            82.0 + state.outbound_level * 180.0,
        ) * state.outbound_level
            * 0.12;

        let left = (mono + inbound * 0.25 - outbound).clamp(-0.25, 0.25) * state.amp;
        let right = (mono + inbound - outbound * 0.25).clamp(-0.25, 0.25) * state.amp;
        write_f32_pair(frame, left, right);
    }
}

fn target_from_spectrum(spectrum: &Spectrum, enabled: bool) -> AudioTarget {
    if spectrum.mirrored {
        let mut inbound = 0.0f32;
        let mut outbound = 0.0f32;
        for pair in spectrum.disp.chunks_exact(2) {
            inbound = inbound.max(pair[0]);
            outbound = outbound.max(pair[1]);
        }

        AudioTarget {
            enabled,
            mono_level: inbound.max(outbound),
            inbound_level: inbound,
            outbound_level: outbound,
            centroid: 0.5,
        }
    } else {
        let mut peak = 0.0f32;
        let mut weighted = 0.0f32;
        let mut total = 0.0f32;
        let bands = spectrum.disp.len().max(1);
        for (i, level) in spectrum.disp.iter().copied().enumerate() {
            peak = peak.max(level);
            total += level;
            weighted += level * i as f32 / bands.saturating_sub(1).max(1) as f32;
        }

        AudioTarget {
            enabled,
            mono_level: peak,
            inbound_level: 0.0,
            outbound_level: 0.0,
            centroid: if total > 0.001 { weighted / total } else { 0.5 },
        }
    }
}

fn soft_tone(phase: &mut f32, hz: f32) -> f32 {
    *phase = (*phase + hz / SAMPLE_RATE).fract();
    (*phase * TAU).sin().tanh()
}

fn write_f32_pair(frame: &mut [u8], left: f32, right: f32) {
    frame[..4].copy_from_slice(&left.to_ne_bytes());
    frame[4..8].copy_from_slice(&right.to_ne_bytes());
}

fn fill_silence(block: &mut [u8]) {
    block.fill(0);
}

#[cfg(test)]
mod tests {
    use crate::bands::{Mode, Spectrum};

    use super::{
        fill_audio_block, target_from_spectrum, write_f32_pair, AudioEngineState, CHANNELS,
    };

    #[test]
    fn non_mirrored_audio_uses_peak_level() {
        let mut spectrum = Spectrum::new(Mode::Protocol, 12);
        spectrum.disp[2] = 0.25;
        spectrum.disp[9] = 0.75;

        let target = target_from_spectrum(&spectrum, true);

        assert!(target.enabled);
        assert_eq!(target.mono_level, 0.75);
        assert_eq!(target.inbound_level, 0.0);
        assert_eq!(target.outbound_level, 0.0);
        assert!(target.centroid > 0.4);
    }

    #[test]
    fn mirrored_audio_keeps_inbound_and_outbound_levels_separate() {
        let mut spectrum = Spectrum::new(Mode::Hybrid, 12);
        spectrum.disp[0] = 0.2;
        spectrum.disp[1] = 0.8;
        spectrum.disp[2] = 0.6;
        spectrum.disp[3] = 0.3;

        let target = target_from_spectrum(&spectrum, true);

        assert_eq!(target.mono_level, 0.8);
        assert_eq!(target.inbound_level, 0.6);
        assert_eq!(target.outbound_level, 0.8);
    }

    #[test]
    fn audio_block_stays_within_safe_range() {
        let mut state = AudioEngineState::default();
        state.target = super::AudioTarget {
            enabled: true,
            mono_level: 1.0,
            inbound_level: 1.0,
            outbound_level: 1.0,
            centroid: 1.0,
        };
        let mut block = vec![0u8; 64 * CHANNELS * std::mem::size_of::<f32>()];

        fill_audio_block(&mut block, &mut state);

        for sample in block.chunks_exact(4) {
            let value = f32::from_ne_bytes(sample.try_into().unwrap());
            assert!((-0.25..=0.25).contains(&value));
        }
    }

    #[test]
    fn writes_native_f32_stereo_frames() {
        let mut frame = [0u8; 8];

        write_f32_pair(&mut frame, 0.5, -0.25);

        assert_eq!(f32::from_ne_bytes(frame[..4].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_ne_bytes(frame[4..].try_into().unwrap()), -0.25);
    }
}
