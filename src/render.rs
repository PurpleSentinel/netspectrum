//! wgpu render loop: builds bar instances from the Spectrum each frame,
//! draws them through shader.wgsl, and overlays text with glyphon.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use crossbeam_channel::Receiver;
use glyphon::{
    Attrs, Buffer as TextBuffer, Color as TextColor, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use crate::audio::{AudioConfig, AudioControl};
use crate::bands::{human_rate, Mode, Spectrum};
use crate::capture::PacketMeta;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// Per-instance payload consumed by shader.wgsl. Every visual is reduced to
// rectangles or particle sprites using this same layout.
struct Inst {
    rect: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

const MAX_INSTANCES: usize = 4096;

// Layout in physical pixels.
const MARGIN_X: f32 = 26.0;
const TOP: f32 = 56.0;
const BOTTOM_LABELS: f32 = 46.0;
const DECORATION_REFRESH_FRAMES: u8 = 4;
const DRAW_MODE_BAR: f32 = 0.0;
const DRAW_MODE_FLAT: f32 = 1.0;
const DRAW_MODE_PARTICLE: f32 = 2.0;
const MAX_FIREWORK_PARTICLES_PER_BAND: usize = 56;
const TRAIL_OVERLAY_ALPHA: f32 = 0.075;

/// Selects which visual builder converts the Spectrum into GPU instances.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VisualStyle {
    Bars,
    Fireworks,
    Radar,
    Matrix,
    Oscilloscope,
    Pulse,
    Galaxy,
    Lightning,
}

/// Colour sets used by particle-based renderers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VisualPalette {
    Neon,
    Solar,
    Aurora,
    Candy,
}

/// Controls how much glyphon text is overlaid on top of the visual.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum HudDetail {
    Full,
    Compact,
    Hidden,
}

/// Physical-pixel plot area shared by all renderers.
#[derive(Copy, Clone)]
struct Plot {
    w: f32,
    h: f32,
    top: f32,
    bottom: f32,
    height: f32,
    slot_w: f32,
}

/// Frame-local switches passed to the selected visual builder.
#[derive(Copy, Clone)]
struct VisualBuildOptions {
    segments: bool,
    transparent_bars: bool,
    style: VisualStyle,
    palette: VisualPalette,
    time: f32,
}

impl VisualPalette {
    /// Cycle through palettes in UI order.
    fn next(self) -> Self {
        match self {
            VisualPalette::Neon => VisualPalette::Solar,
            VisualPalette::Solar => VisualPalette::Aurora,
            VisualPalette::Aurora => VisualPalette::Candy,
            VisualPalette::Candy => VisualPalette::Neon,
        }
    }

    /// Short label shown in the HUD.
    fn name(self) -> &'static str {
        match self {
            VisualPalette::Neon => "neon",
            VisualPalette::Solar => "solar",
            VisualPalette::Aurora => "aurora",
            VisualPalette::Candy => "candy",
        }
    }
}

impl HudDetail {
    /// Cycle full -> compact -> hidden -> full.
    fn next(self) -> Self {
        match self {
            HudDetail::Full => HudDetail::Compact,
            HudDetail::Compact => HudDetail::Hidden,
            HudDetail::Hidden => HudDetail::Full,
        }
    }

    /// Short label shown in the HUD.
    fn name(self) -> &'static str {
        match self {
            HudDetail::Full => "full",
            HudDetail::Compact => "mini",
            HudDetail::Hidden => "clean",
        }
    }

    /// Hidden mode suppresses all text preparation and rendering.
    fn shows_header(self) -> bool {
        !matches!(self, HudDetail::Hidden)
    }

    /// Only full mode shows per-band labels and live rates.
    fn shows_labels(self) -> bool {
        matches!(self, HudDetail::Full)
    }
}

/// Run the winit/wgpu application loop until the window exits.
pub fn run(
    mut spectrum: Spectrum,
    rx: Receiver<PacketMeta>,
    iface: String,
    audio_config: AudioConfig,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("netspectrum")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_transparent(true)
            .build(&event_loop)?,
    );

    // ------------------------------------------------------------- wgpu setup
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone())?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("no compatible GPU adapter found");

    let adapter_limits = adapter.limits();
    let max_surface_extent = adapter_limits.max_texture_dimension_2d;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
        },
        None,
    ))?;

    let size = window.inner_size();
    let (surface_width, surface_height) =
        surface_extent(size.width, size.height, max_surface_extent);
    let mut config = surface
        .get_default_config(&adapter, surface_width, surface_height)
        .expect("surface unsupported by adapter");
    config.present_mode = wgpu::PresentMode::AutoVsync;
    config.alpha_mode = preferred_alpha_mode(&surface.get_capabilities(&adapter).alpha_modes);
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bars"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bars"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Inst>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: (MAX_INSTANCES * std::mem::size_of::<Inst>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ------------------------------------------------------------- text setup
    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();
    let mut atlas = TextAtlas::new(&device, &queue, config.format);
    let mut text_renderer =
        TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);

    let mut header_buf = TextBuffer::new(&mut font_system, Metrics::new(15.0, 20.0));
    let mut label_bufs: Vec<TextBuffer> = Vec::new();

    // ------------------------------------------------------------- state
    let mut segments = true;
    let mut transparent_background = false;
    let mut transparent_bars = false;
    let mut window_decorated = true;
    let mut decoration_refresh_frames = 0u8;
    let mut visual_style = VisualStyle::Bars;
    let mut visual_palette = VisualPalette::Neon;
    let mut hud_detail = HudDetail::Full;
    let mut audio = AudioControl::new(audio_config);
    let mut last = Instant::now();
    let mut animation_time = 0.0f32;
    let mut label_refresh = 0.0f32;
    let mut instances: Vec<Inst> = Vec::with_capacity(MAX_INSTANCES);

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    // Clamp to adapter limits before configuring the swapchain;
                    // some drivers reject oversized transparent windows.
                    (config.width, config.height) =
                        surface_extent(new_size.width, new_size.height, max_surface_extent);
                    surface.configure(&device, &config);
                    spectrum.labels_dirty = true;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // All interactive controls are handled here so renderer state
                // changes are synchronized with the next redraw.
                match code {
                    KeyCode::Digit1 => spectrum.set_mode(Mode::Protocol),
                    KeyCode::Digit2 => spectrum.set_mode(Mode::Ports),
                    KeyCode::Digit3 => spectrum.set_mode(Mode::Hosts),
                    KeyCode::Digit4 => spectrum.set_mode(Mode::Hybrid),
                    KeyCode::Digit5 => spectrum.set_mode(Mode::Sizes),
                    KeyCode::KeyS => segments = !segments,
                    KeyCode::KeyB => visual_style = VisualStyle::Bars,
                    KeyCode::KeyF => visual_style = VisualStyle::Fireworks,
                    KeyCode::KeyR => visual_style = VisualStyle::Radar,
                    KeyCode::KeyM => visual_style = VisualStyle::Matrix,
                    KeyCode::KeyO => visual_style = VisualStyle::Oscilloscope,
                    KeyCode::KeyP => visual_style = VisualStyle::Pulse,
                    KeyCode::KeyG => visual_style = VisualStyle::Galaxy,
                    KeyCode::KeyL => visual_style = VisualStyle::Lightning,
                    KeyCode::KeyY => visual_palette = visual_palette.next(),
                    KeyCode::KeyH => hud_detail = hud_detail.next(),
                    KeyCode::KeyW => {
                        window_decorated = next_window_decoration_request(window.is_decorated());
                        decoration_refresh_frames = DECORATION_REFRESH_FRAMES;
                        apply_window_decorations(&window, window_decorated);
                    }
                    KeyCode::KeyZ => transparent_background = !transparent_background,
                    KeyCode::KeyX => transparent_bars = !transparent_bars,
                    KeyCode::KeyA => {
                        audio.toggle();
                    }
                    KeyCode::KeyT => {
                        audio.cycle_palette();
                    }
                    KeyCode::KeyQ | KeyCode::Escape => elwt.exit(),
                    _ => {}
                };
            }
            WindowEvent::RedrawRequested => {
                if decoration_refresh_frames > 0 {
                    // Some window managers need a few frames of repeated
                    // decoration requests before the frame state sticks.
                    apply_window_decorations(&window, window_decorated);
                    decoration_refresh_frames -= 1;
                }

                let now = Instant::now();
                let dt = (now - last).as_secs_f32();
                last = now;
                animation_time += dt;

                // Drain pending packet metadata for this frame, then run one
                // signal-processing tick before building GPU instances.
                for m in rx.try_iter() {
                    spectrum.ingest(&m);
                }
                spectrum.tick(dt);
                audio.update(&spectrum);

                label_refresh -= dt;
                if spectrum.labels_dirty || label_refresh <= 0.0 {
                    // Re-shaping text is relatively expensive, so refresh
                    // labels at a modest cadence unless the mode changed.
                    rebuild_labels(&mut font_system, &mut label_bufs, &spectrum, &config);
                    spectrum.labels_dirty = false;
                    label_refresh = 0.25;
                }

                let w = config.width as f32;
                let h = config.height as f32;
                build_visual_instances(
                    &mut instances,
                    &spectrum,
                    w,
                    h,
                    VisualBuildOptions {
                        segments,
                        transparent_bars,
                        style: visual_style,
                        palette: visual_palette,
                        time: animation_time,
                    },
                );
                queue.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&instances));

                // Header text is rebuilt every frame because throughput and
                // live toggle labels are dynamic.
                let audio_snapshot = audio.snapshot();
                let audio_label = if audio_snapshot.enabled {
                    "audio on"
                } else if audio_snapshot.available {
                    "audio off"
                } else {
                    "audio n/a"
                };
                let tone_label = audio_snapshot.palette.name();
                let background_label = if transparent_background { "bg clr" } else { "bg on" };
                let bars_label = if transparent_bars { "bar clr" } else { "bar gry" };
                let frame_label = window_frame_label(window_decorated);
                let visual_label = visual_style_label(visual_style);
                let palette_label = visual_palette.name();
                let hud_label = hud_detail.name();
                let header = if hud_detail.shows_header() {
                    match hud_detail {
                        HudDetail::Full => Some(format!(
                            "NETSPECTRUM {} [{}]  in {}  out {}   1-5 modes  V {}  Y {}  H {}  S seg  A {}  T {}  W {}  Z {}  X {}  Q quit",
                            iface,
                            spectrum.mode.name(),
                            human_rate(spectrum.rate_in),
                            human_rate(spectrum.rate_out),
                            visual_label,
                            palette_label,
                            hud_label,
                            audio_label,
                            tone_label,
                            frame_label,
                            background_label,
                            bars_label,
                        )),
                        HudDetail::Compact => Some(format!(
                            "NETSPECTRUM {} [{}]  in {}  out {}  V {}  Y {}  H {}  Q quit",
                            iface,
                            spectrum.mode.name(),
                            human_rate(spectrum.rate_in),
                            human_rate(spectrum.rate_out),
                            visual_label,
                            palette_label,
                            hud_label,
                        )),
                        HudDetail::Hidden => None,
                    }
                } else {
                    None
                };

                let mut areas: Vec<TextArea> = Vec::new();
                if let Some(header) = header {
                    header_buf.set_size(&mut font_system, w, 40.0);
                    header_buf.set_text(
                        &mut font_system,
                        &header,
                        Attrs::new().family(Family::Monospace),
                        Shaping::Advanced,
                    );
                    areas.push(TextArea {
                        buffer: &header_buf,
                        left: MARGIN_X,
                        top: 16.0,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: 0,
                            top: 0,
                            right: config.width as i32,
                            bottom: config.height as i32,
                        },
                        default_color: TextColor::rgb(225, 232, 245),
                    });
                }

                let n = spectrum.band_count().max(1) as f32;
                let slot_w = (w - 2.0 * MARGIN_X) / n;
                if hud_detail.shows_labels() {
                    for (i, buf) in label_bufs.iter().enumerate() {
                        let x = MARGIN_X + i as f32 * slot_w;
                        areas.push(TextArea {
                            buffer: buf,
                            left: x,
                            top: h - BOTTOM_LABELS + 4.0,
                            scale: 1.0,
                            bounds: TextBounds {
                                left: x as i32,
                                top: 0,
                                right: (x + slot_w) as i32,
                                bottom: config.height as i32,
                            },
                            default_color: TextColor::rgb(150, 160, 178),
                        });
                    }
                }

                if text_renderer
                    .prepare(
                        &device,
                        &queue,
                        &mut font_system,
                        &mut atlas,
                        Resolution {
                            width: config.width,
                            height: config.height,
                        },
                        areas,
                        &mut swash_cache,
                    )
                    .is_err()
                {
                    // Text failure should never kill the visualiser.
                }

                // Draw visual instances first, then glyphon text in the same
                // render pass using alpha blending.
                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                        surface.configure(&device, &config);
                        return;
                    }
                    Err(_) => return,
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(background_color(
                                    transparent_background,
                                )),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_vertex_buffer(0, inst_buf.slice(..));
                    pass.draw(0..6, 0..instances.len() as u32);
                    let _ = text_renderer.render(&atlas, &mut pass);
                }
                queue.submit(Some(encoder.finish()));
                frame.present();
                atlas.trim();
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    })?;
    Ok(())
}

fn surface_extent(width: u32, height: u32, max_texture_dimension_2d: u32) -> (u32, u32) {
    let max_extent = max_texture_dimension_2d.max(1);
    (width.clamp(1, max_extent), height.clamp(1, max_extent))
}

/// Prefer alpha-capable surface modes so the background transparency toggle can
/// work on compositors that support it.
fn preferred_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> wgpu::CompositeAlphaMode {
    [
        wgpu::CompositeAlphaMode::PreMultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied,
        wgpu::CompositeAlphaMode::Inherit,
        wgpu::CompositeAlphaMode::Auto,
        wgpu::CompositeAlphaMode::Opaque,
    ]
    .into_iter()
    .find(|mode| modes.contains(mode))
    .unwrap_or(wgpu::CompositeAlphaMode::Opaque)
}

/// Swapchain clear colour. Transparent mode keeps RGB black with zero alpha.
fn background_color(transparent: bool) -> wgpu::Color {
    if transparent {
        return wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
    }

    wgpu::Color {
        r: 0.015,
        g: 0.022,
        b: 0.038,
        a: 1.0,
    }
}

/// HUD label for the current window decoration state.
fn window_frame_label(decorated: bool) -> &'static str {
    if decorated {
        "frame"
    } else {
        "bare"
    }
}

/// Compute the next requested decoration state from the current window state.
fn next_window_decoration_request(currently_decorated: bool) -> bool {
    !currently_decorated
}

/// Apply frame/titlebar visibility and ask the compositor to preserve size.
fn apply_window_decorations(window: &Window, decorated: bool) {
    let inner_size = window.inner_size();
    window.set_decorations(decorated);
    let _ = window.request_inner_size(inner_size);
    window.set_visible(true);
    window.request_redraw();
}

/// Short HUD label for the selected visual renderer.
fn visual_style_label(style: VisualStyle) -> &'static str {
    match style {
        VisualStyle::Bars => "bars",
        VisualStyle::Fireworks => "fire",
        VisualStyle::Radar => "radar",
        VisualStyle::Matrix => "matrix",
        VisualStyle::Oscilloscope => "scope",
        VisualStyle::Pulse => "pulse",
        VisualStyle::Galaxy => "galaxy",
        VisualStyle::Lightning => "bolt",
    }
}

/// Dispatch to the active renderer while keeping one shared output buffer.
fn build_visual_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    options: VisualBuildOptions,
) {
    match options.style {
        VisualStyle::Bars => build_instances(
            out,
            spectrum,
            w,
            h,
            options.segments,
            options.transparent_bars,
        ),
        VisualStyle::Fireworks => {
            build_firework_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Radar => {
            build_radar_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Matrix => {
            build_matrix_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Oscilloscope => {
            build_oscilloscope_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Pulse => {
            build_pulse_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Galaxy => {
            build_galaxy_instances(out, spectrum, w, h, options.time, options.palette)
        }
        VisualStyle::Lightning => {
            build_lightning_instances(out, spectrum, w, h, options.time, options.palette)
        }
    }
}

/// Compute the drawable plot rectangle used by particle renderers.
fn plot(w: f32, h: f32, band_count: usize) -> Plot {
    let top = TOP + 6.0;
    let bottom = h - BOTTOM_LABELS - 4.0;
    let height = (bottom - top).max(1.0);
    let slot_w = (w - 2.0 * MARGIN_X) / band_count.max(1) as f32;
    Plot {
        w,
        h,
        top,
        bottom,
        height,
        slot_w,
    }
}

/// Convert physical pixel x-coordinate into normalized device coordinates.
fn ndc_x(px: f32, w: f32) -> f32 {
    px / w * 2.0 - 1.0
}

/// Convert physical pixel y-coordinate into normalized device coordinates.
fn ndc_y(py: f32, h: f32) -> f32 {
    1.0 - py / h * 2.0
}

/// Push one instanced rectangle after converting from pixels to NDC.
fn push_instance(
    out: &mut Vec<Inst>,
    plot: Plot,
    rect: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
) {
    if out.len() < MAX_INSTANCES {
        out.push(Inst {
            rect: [
                ndc_x(rect[0], plot.w),
                ndc_y(rect[1], plot.h),
                ndc_x(rect[2], plot.w),
                ndc_y(rect[3], plot.h),
            ],
            color,
            params,
        });
    }
}

/// Push one circular soft particle sprite, represented as a square instance.
fn push_particle(
    out: &mut Vec<Inst>,
    plot: Plot,
    cx: f32,
    cy: f32,
    size: f32,
    color: [f32; 4],
    intensity: f32,
) {
    let r = size * 0.5;
    push_instance(
        out,
        plot,
        [cx - r, cy + r, cx + r, cy - r],
        color,
        [intensity, DRAW_MODE_PARTICLE, 0.0, 0.0],
    );
}

/// Push a faint oversized glow sprite plus the main particle.
fn push_bloom_particle(
    out: &mut Vec<Inst>,
    plot: Plot,
    cx: f32,
    cy: f32,
    size: f32,
    color: [f32; 4],
    intensity: f32,
) {
    let glow = [color[0], color[1], color[2], color[3] * 0.22];
    push_particle(out, plot, cx, cy, size * 2.6, glow, intensity);
    push_particle(out, plot, cx, cy, size, color, intensity);
}

/// Particle plus its motion vector used for deterministic blur trails.
struct TrailParticle {
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    size: f32,
    color: [f32; 4],
    intensity: f32,
    steps: usize,
}

/// Draw fading ghost particles behind a moving bloom particle.
fn push_trail_bloom_particle(out: &mut Vec<Inst>, plot: Plot, particle: TrailParticle) {
    let steps = particle.steps.min(5);
    for step in (1..=steps).rev() {
        let k = step as f32 / (steps + 1) as f32;
        let trail_color = [
            particle.color[0],
            particle.color[1],
            particle.color[2],
            particle.color[3] * (0.08 + (1.0 - k) * 0.14),
        ];
        push_particle(
            out,
            plot,
            particle.cx - particle.dx * k,
            particle.cy - particle.dy * k,
            particle.size * (0.64 + (1.0 - k) * 0.28),
            trail_color,
            particle.intensity,
        );
    }

    push_bloom_particle(
        out,
        plot,
        particle.cx,
        particle.cy,
        particle.size,
        particle.color,
        particle.intensity,
    );
}

/// Push a flat-colour rectangle for caps, baselines, and overlays.
fn push_flat_rect(
    out: &mut Vec<Inst>,
    plot: Plot,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    push_instance(
        out,
        plot,
        [x0, y0, x1, y1],
        color,
        [0.0, DRAW_MODE_FLAT, 0.0, 0.0],
    );
}

/// Return a single activity level per labelled band.
fn band_level(spectrum: &Spectrum, band: usize) -> f32 {
    if spectrum.mirrored {
        let i = band * 2;
        spectrum.disp[i].max(spectrum.disp[i + 1]).clamp(0.0, 1.0)
    } else {
        spectrum.disp[band].clamp(0.0, 1.0)
    }
}

/// Return inbound/outbound levels for renderers that care about direction.
fn band_signed_levels(spectrum: &Spectrum, band: usize) -> (f32, f32) {
    if spectrum.mirrored {
        (
            spectrum.disp[band * 2].clamp(0.0, 1.0),
            spectrum.disp[band * 2 + 1].clamp(0.0, 1.0),
        )
    } else {
        (spectrum.disp[band].clamp(0.0, 1.0), 0.0)
    }
}

/// Pick the palette colour assigned to a band index.
fn palette_tint(palette: VisualPalette, band: usize) -> [f32; 3] {
    const NEON: [[f32; 3]; 8] = [
        [0.05, 1.0, 0.95],
        [1.0, 0.92, 0.05],
        [1.0, 0.08, 0.45],
        [0.42, 0.18, 1.0],
        [0.22, 1.0, 0.18],
        [1.0, 0.36, 0.02],
        [0.2, 0.62, 1.0],
        [1.0, 0.18, 0.95],
    ];
    const SOLAR: [[f32; 3]; 8] = [
        [1.0, 0.82, 0.16],
        [1.0, 0.42, 0.04],
        [1.0, 0.16, 0.08],
        [1.0, 0.62, 0.0],
        [0.95, 0.96, 0.28],
        [1.0, 0.24, 0.02],
        [1.0, 0.72, 0.10],
        [0.9, 0.34, 0.06],
    ];
    const AURORA: [[f32; 3]; 8] = [
        [0.08, 1.0, 0.54],
        [0.02, 0.82, 1.0],
        [0.36, 0.44, 1.0],
        [0.10, 1.0, 0.92],
        [0.58, 1.0, 0.18],
        [0.28, 0.92, 1.0],
        [0.68, 0.34, 1.0],
        [0.06, 1.0, 0.70],
    ];
    const CANDY: [[f32; 3]; 8] = [
        [1.0, 0.24, 0.72],
        [0.72, 0.24, 1.0],
        [1.0, 0.55, 0.88],
        [0.36, 0.80, 1.0],
        [1.0, 0.88, 0.22],
        [0.98, 0.34, 1.0],
        [0.28, 1.0, 0.88],
        [1.0, 0.38, 0.56],
    ];

    let colors = match palette {
        VisualPalette::Neon => NEON,
        VisualPalette::Solar => SOLAR,
        VisualPalette::Aurora => AURORA,
        VisualPalette::Candy => CANDY,
    };
    colors[band % colors.len()]
}

/// Add a low-alpha overlay under particle renderers to visually soften trails.
fn push_trail_overlay(out: &mut Vec<Inst>, plot: Plot) {
    push_flat_rect(
        out,
        plot,
        0.0,
        plot.h,
        plot.w,
        0.0,
        [0.0, 0.0, 0.0, TRAIL_OVERLAY_ALPHA],
    );
}

/// Scale firework particle count with band intensity.
fn firework_particle_count(intensity: f32) -> usize {
    if intensity <= 0.002 {
        return 0;
    }

    (10.0 + intensity.clamp(0.0, 1.0) * 46.0).round() as usize
}

/// Small deterministic hash used for stable pseudo-random visual placement.
fn hash01(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

/// Combine palette tint, intensity, and per-particle sparkle into RGBA.
fn firework_color(intensity: f32, seed: u32, tint: [f32; 3], alpha: f32) -> [f32; 4] {
    let sparkle = hash01(seed);
    let hot = intensity.clamp(0.0, 1.0);
    [
        (tint[0] * (0.95 + 0.35 * hot) + sparkle * 0.25).min(1.0),
        (tint[1] * (0.92 + 0.38 * hot) + sparkle * 0.23).min(1.0),
        (tint[2] * (0.90 + 0.40 * hot) + sparkle * 0.22).min(1.0),
        alpha.clamp(0.0, 1.0),
    ]
}

/// Rebuild glyphon buffers for per-band labels and rates.
fn rebuild_labels(
    font_system: &mut FontSystem,
    label_bufs: &mut Vec<TextBuffer>,
    spectrum: &Spectrum,
    config: &wgpu::SurfaceConfiguration,
) {
    let n = spectrum.band_count();
    let w = config.width as f32;
    let slot_w = ((w - 2.0 * MARGIN_X) / n.max(1) as f32).max(8.0);
    let font_px = if n > 14 { 10.0 } else { 12.0 };

    label_bufs.clear();
    for i in 0..n {
        let mut buf = TextBuffer::new(font_system, Metrics::new(font_px, font_px + 3.0));
        buf.set_size(font_system, slot_w, 40.0);
        let text = format!(
            "{}\n{}",
            spectrum.labels[i],
            human_rate(spectrum.band_rate(i))
        );
        buf.set_text(
            font_system,
            &text,
            Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
        );
        for line in buf.lines.iter_mut() {
            line.set_align(Some(glyphon::cosmic_text::Align::Center));
        }
        buf.shape_until_scroll(font_system);
        label_bufs.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        background_color, build_firework_instances, build_galaxy_instances, build_instances,
        build_lightning_instances, build_matrix_instances, build_oscilloscope_instances,
        build_pulse_instances, build_radar_instances, build_visual_instances,
        firework_particle_count, next_window_decoration_request, palette_tint,
        preferred_alpha_mode, surface_extent, visual_style_label, window_frame_label, HudDetail,
        Inst, VisualBuildOptions, VisualPalette, VisualStyle, DRAW_MODE_PARTICLE, MAX_INSTANCES,
    };
    use crate::bands::{Mode, Spectrum};

    #[test]
    fn surface_extent_clamps_to_device_limit() {
        assert_eq!(surface_extent(2060, 669, 2048), (2048, 669));
    }

    #[test]
    fn surface_extent_never_returns_zero() {
        assert_eq!(surface_extent(0, 0, 2048), (1, 1));
        assert_eq!(surface_extent(10, 10, 0), (1, 1));
    }

    #[test]
    fn background_color_alpha_can_be_toggled() {
        assert_eq!(background_color(false).a, 1.0);
        assert_eq!(background_color(true).a, 0.0);
    }

    #[test]
    fn window_frame_label_tracks_decoration_state() {
        assert_eq!(window_frame_label(true), "frame");
        assert_eq!(window_frame_label(false), "bare");
    }

    #[test]
    fn visual_style_label_tracks_selected_renderer() {
        assert_eq!(visual_style_label(VisualStyle::Bars), "bars");
        assert_eq!(visual_style_label(VisualStyle::Fireworks), "fire");
        assert_eq!(visual_style_label(VisualStyle::Radar), "radar");
        assert_eq!(visual_style_label(VisualStyle::Matrix), "matrix");
        assert_eq!(visual_style_label(VisualStyle::Oscilloscope), "scope");
        assert_eq!(visual_style_label(VisualStyle::Pulse), "pulse");
        assert_eq!(visual_style_label(VisualStyle::Galaxy), "galaxy");
        assert_eq!(visual_style_label(VisualStyle::Lightning), "bolt");
    }

    #[test]
    fn visual_palette_cycles_and_names_are_stable() {
        assert_eq!(VisualPalette::Neon.name(), "neon");
        assert_eq!(VisualPalette::Neon.next(), VisualPalette::Solar);
        assert_eq!(VisualPalette::Solar.next(), VisualPalette::Aurora);
        assert_eq!(VisualPalette::Aurora.next(), VisualPalette::Candy);
        assert_eq!(VisualPalette::Candy.next(), VisualPalette::Neon);
    }

    #[test]
    fn visual_palettes_produce_distinct_tints() {
        assert_ne!(
            palette_tint(VisualPalette::Neon, 0),
            palette_tint(VisualPalette::Solar, 0)
        );
        assert_ne!(
            palette_tint(VisualPalette::Aurora, 3),
            palette_tint(VisualPalette::Candy, 3)
        );
    }

    #[test]
    fn hud_detail_cycles_from_full_to_clean_and_back() {
        assert_eq!(HudDetail::Full.name(), "full");
        assert!(HudDetail::Full.shows_header());
        assert!(HudDetail::Full.shows_labels());

        assert_eq!(HudDetail::Full.next(), HudDetail::Compact);
        assert_eq!(HudDetail::Compact.name(), "mini");
        assert!(HudDetail::Compact.shows_header());
        assert!(!HudDetail::Compact.shows_labels());

        assert_eq!(HudDetail::Compact.next(), HudDetail::Hidden);
        assert_eq!(HudDetail::Hidden.name(), "clean");
        assert!(!HudDetail::Hidden.shows_header());
        assert!(!HudDetail::Hidden.shows_labels());
        assert_eq!(HudDetail::Hidden.next(), HudDetail::Full);
    }

    #[test]
    fn shader_wgsl_parses() {
        naga::front::wgsl::parse_str(include_str!("shader.wgsl")).expect("shader should parse");
    }

    #[test]
    fn next_window_decoration_request_toggles_current_state() {
        assert!(!next_window_decoration_request(true));
        assert!(next_window_decoration_request(false));
    }

    #[test]
    fn preferred_alpha_mode_allows_transparency_when_supported() {
        assert_eq!(
            preferred_alpha_mode(&[
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PostMultiplied,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ]),
            wgpu::CompositeAlphaMode::PreMultiplied
        );
        assert_eq!(
            preferred_alpha_mode(&[wgpu::CompositeAlphaMode::Opaque]),
            wgpu::CompositeAlphaMode::Opaque
        );
    }

    #[test]
    fn ghost_bar_alpha_can_be_toggled() {
        let spectrum = Spectrum::new(Mode::Protocol, 12);
        let mut instances = Vec::new();

        build_instances(&mut instances, &spectrum, 1280.0, 720.0, true, false);
        assert!(instances.iter().any(|inst| inst.color[3] > 0.0));

        build_instances(&mut instances, &spectrum, 1280.0, 720.0, true, true);
        assert!(instances.iter().all(|inst| inst.color[3] == 0.0));
    }

    #[test]
    fn firework_particle_count_scales_with_intensity() {
        assert_eq!(firework_particle_count(0.0), 0);
        assert!(firework_particle_count(0.15) >= 16);
        assert!(firework_particle_count(0.9) > firework_particle_count(0.2));
        assert!(firework_particle_count(1.0) >= 50);
    }

    #[test]
    fn fireworks_emit_more_particles_for_louder_bands() {
        let mut spectrum = Spectrum::new(Mode::Protocol, 12);
        let mut quiet = Vec::new();
        spectrum.disp[0] = 0.15;
        build_firework_instances(
            &mut quiet,
            &spectrum,
            1280.0,
            720.0,
            0.25,
            VisualPalette::Neon,
        );

        let mut loud = Vec::new();
        spectrum.disp[0] = 0.90;
        build_firework_instances(
            &mut loud,
            &spectrum,
            1280.0,
            720.0,
            0.25,
            VisualPalette::Neon,
        );

        assert!(loud.len() > quiet.len());
        assert!(loud.iter().any(|inst| inst.params[1] == DRAW_MODE_PARTICLE));
    }

    #[test]
    fn every_visual_style_emits_instances_for_active_spectrum() {
        let mut spectrum = Spectrum::new(Mode::Protocol, 12);
        spectrum.disp[0] = 0.75;
        spectrum.disp[1] = 0.35;

        for style in [
            VisualStyle::Bars,
            VisualStyle::Fireworks,
            VisualStyle::Radar,
            VisualStyle::Matrix,
            VisualStyle::Oscilloscope,
            VisualStyle::Pulse,
            VisualStyle::Galaxy,
            VisualStyle::Lightning,
        ] {
            let mut instances = Vec::new();
            build_visual_instances(
                &mut instances,
                &spectrum,
                1280.0,
                720.0,
                VisualBuildOptions {
                    segments: true,
                    transparent_bars: false,
                    style,
                    palette: VisualPalette::Neon,
                    time: 0.33,
                },
            );
            assert!(!instances.is_empty(), "{style:?} should draw instances");
            assert!(instances.len() <= MAX_INSTANCES);
        }
    }

    #[test]
    fn supplemental_visual_builders_emit_particles() {
        let mut spectrum = Spectrum::new(Mode::Hybrid, 12);
        spectrum.disp[0] = 0.6;
        spectrum.disp[1] = 0.4;

        for build in [
            build_radar_instances as fn(&mut Vec<Inst>, &Spectrum, f32, f32, f32, VisualPalette),
            build_matrix_instances,
            build_oscilloscope_instances,
            build_pulse_instances,
            build_galaxy_instances,
            build_lightning_instances,
        ] {
            let mut instances = Vec::new();
            build(
                &mut instances,
                &spectrum,
                1280.0,
                720.0,
                0.45,
                VisualPalette::Neon,
            );
            assert!(instances
                .iter()
                .any(|inst| inst.params[1] == DRAW_MODE_PARTICLE));
        }
    }
}

fn build_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    segments: bool,
    transparent_bars: bool,
) {
    out.clear();
    // Classic renderer: each band becomes a ghost slot, optional lit bar, and
    // optional peak cap.
    let ndc_x = |px: f32| px / w * 2.0 - 1.0;
    let ndc_y = |py: f32| 1.0 - py / h * 2.0;

    let plot_top = TOP;
    let plot_bottom = h - BOTTOM_LABELS;
    let plot_h = (plot_bottom - plot_top).max(1.0);
    let n = spectrum.band_count().max(1) as f32;
    let slot_w = (w - 2.0 * MARGIN_X) / n;
    let bar_pad = slot_w * 0.14;
    let seg_flag = if segments { 1.0 } else { 0.0 };

    let white = [1.0, 1.0, 1.0, 0.95];
    let tint_in = [0.62, 0.92, 1.0, 0.95];
    let tint_out = [1.0, 0.72, 0.5, 0.95];
    let ghost_alpha = if transparent_bars { 0.0 } else { 0.045 };
    let ghost = [1.0, 1.0, 1.0, ghost_alpha];
    let peak_col = [1.0, 0.88, 0.35, 0.9];

    let push = |out: &mut Vec<Inst>,
                x0: f32,
                y0: f32,
                x1: f32,
                y1: f32,
                color: [f32; 4],
                params: [f32; 4]| {
        if out.len() < MAX_INSTANCES {
            out.push(Inst {
                rect: [ndc_x(x0), ndc_y(y0), ndc_x(x1), ndc_y(y1)],
                color,
                params,
            });
        }
    };

    if !spectrum.mirrored {
        for i in 0..spectrum.band_count() {
            let x0 = MARGIN_X + i as f32 * slot_w + bar_pad;
            let x1 = MARGIN_X + (i as f32 + 1.0) * slot_w - bar_pad;

            // Ghost slot (full scale, faint).
            push(
                out,
                x0,
                plot_bottom,
                x1,
                plot_top,
                ghost,
                [1.0, DRAW_MODE_FLAT, 0.0, 0.0],
            );

            let v = spectrum.disp[i];
            if v > 0.002 {
                let tip = plot_bottom - v * plot_h;
                push(
                    out,
                    x0,
                    plot_bottom,
                    x1,
                    tip,
                    white,
                    [v, DRAW_MODE_BAR, seg_flag, 0.0],
                );
            }
            let p = spectrum.peak[i];
            if p > 0.004 {
                let py = plot_bottom - p * plot_h;
                push(
                    out,
                    x0,
                    py + 1.5,
                    x1,
                    py - 1.5,
                    peak_col,
                    [p, DRAW_MODE_FLAT, 0.0, 0.0],
                );
            }
        }
    } else {
        // Mirrored hybrid: inbound rises from the centre line, outbound falls.
        let mid = (plot_top + plot_bottom) * 0.5;
        let half_h = plot_h * 0.5 - 2.0;
        for g in 0..spectrum.band_count() {
            let x0 = MARGIN_X + g as f32 * slot_w + bar_pad;
            let x1 = MARGIN_X + (g as f32 + 1.0) * slot_w - bar_pad;

            push(
                out,
                x0,
                plot_bottom,
                x1,
                plot_top,
                ghost,
                [1.0, DRAW_MODE_FLAT, 0.0, 0.0],
            );
            // Centre line tick.
            push(
                out,
                x0,
                mid + 1.0,
                x1,
                mid - 1.0,
                [1.0, 1.0, 1.0, 0.16],
                [0.0, DRAW_MODE_FLAT, 0.0, 0.0],
            );

            let up = spectrum.disp[g * 2];
            if up > 0.002 {
                let tip = mid - up * half_h;
                push(
                    out,
                    x0,
                    mid,
                    x1,
                    tip,
                    tint_in,
                    [up, DRAW_MODE_BAR, seg_flag, 0.0],
                );
            }
            let pu = spectrum.peak[g * 2];
            if pu > 0.004 {
                let py = mid - pu * half_h;
                push(
                    out,
                    x0,
                    py + 1.5,
                    x1,
                    py - 1.5,
                    peak_col,
                    [pu, DRAW_MODE_FLAT, 0.0, 0.0],
                );
            }

            let down = spectrum.disp[g * 2 + 1];
            if down > 0.002 {
                let tip = mid + down * half_h;
                push(
                    out,
                    x0,
                    mid,
                    x1,
                    tip,
                    tint_out,
                    [down, DRAW_MODE_BAR, seg_flag, 0.0],
                );
            }
            let pd = spectrum.peak[g * 2 + 1];
            if pd > 0.004 {
                let py = mid + pd * half_h;
                push(
                    out,
                    x0,
                    py - 1.5,
                    x1,
                    py + 1.5,
                    peak_col,
                    [pd, DRAW_MODE_FLAT, 0.0, 0.0],
                );
            }
        }
    }
}

fn build_firework_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);

    // Fireworks reuse a closure because non-mirrored and mirrored modes differ
    // only in where the burst starts and which direction it travels.
    let emit_firework = |out: &mut Vec<Inst>,
                         band: usize,
                         cx: f32,
                         base_y: f32,
                         burst_y: f32,
                         intensity: f32,
                         tint: [f32; 3]| {
        let count = firework_particle_count(intensity);
        if count == 0 {
            return;
        }

        let phase = (time * (0.48 + intensity * 1.25) + band as f32 * 0.137).fract();
        let fade = (1.0 - phase).powf(0.75);
        let radius =
            (p.slot_w * (0.22 + intensity * 1.05)).min(p.height * 0.30) * (0.18 + phase * 1.25);
        let seed_base = (band as u32 + 1).wrapping_mul(7919);

        let trail_count = (5.0 + intensity * 11.0).round() as usize;
        for trail in 0..trail_count {
            let k = (trail + 1) as f32 / (trail_count + 1) as f32;
            let wobble = (hash01(seed_base ^ trail as u32) - 0.5) * p.slot_w * 0.16;
            let y = base_y + (burst_y - base_y) * k;
            let alpha = (0.12 + intensity * 0.32) * (1.0 - k * 0.40);
            let color = firework_color(intensity, seed_base ^ trail as u32, tint, alpha);
            push_bloom_particle(
                out,
                p,
                cx + wobble,
                y,
                4.0 + intensity * 7.0,
                color,
                intensity,
            );
        }

        let core_alpha = (0.28 + intensity * 0.72) * fade;
        let core_size = 11.0 + intensity * 38.0 * (1.0 - phase * 0.25);
        push_bloom_particle(
            out,
            p,
            cx,
            burst_y,
            core_size,
            firework_color(intensity, seed_base ^ 0xa5a5, tint, core_alpha),
            intensity,
        );

        for i in 0..count.min(MAX_FIREWORK_PARTICLES_PER_BAND) {
            let seed = seed_base ^ ((i as u32 + 3).wrapping_mul(104_729));
            let angle =
                std::f32::consts::TAU * (i as f32 / count as f32 + (hash01(seed) - 0.5) * 0.10);
            let scatter = 0.78 + hash01(seed ^ 0x3456) * 0.46;
            let gravity = phase * phase * radius * 0.42;
            let px = cx + angle.cos() * radius * scatter;
            let py = burst_y + angle.sin() * radius * scatter + gravity;
            let size = 4.5 + intensity * 13.0 * (1.0 - phase * 0.18);
            let alpha = fade * (0.32 + intensity * 0.88) * (0.82 + hash01(seed ^ 0x789a) * 0.34);
            let color = firework_color(intensity, seed, tint, alpha);
            push_trail_bloom_particle(
                out,
                p,
                TrailParticle {
                    cx: px,
                    cy: py,
                    dx: (px - cx) * 0.22,
                    dy: (py - burst_y) * 0.22 + gravity * 0.18,
                    size,
                    color,
                    intensity,
                    steps: 4,
                },
            );

            if i % 2 == 0 {
                let inner_px = cx + angle.cos() * radius * scatter * 0.45;
                let inner_py = burst_y + angle.sin() * radius * scatter * 0.45 + gravity * 0.35;
                let inner_alpha = alpha * (0.68 + intensity * 0.24);
                push_trail_bloom_particle(
                    out,
                    p,
                    TrailParticle {
                        cx: inner_px,
                        cy: inner_py,
                        dx: (inner_px - cx) * 0.26,
                        dy: (inner_py - burst_y) * 0.26 + gravity * 0.10,
                        size: size * 0.72,
                        color: firework_color(intensity, seed ^ 0xd00d, tint, inner_alpha),
                        intensity,
                        steps: 3,
                    },
                );
            }
        }
    };

    if !spectrum.mirrored {
        for band in 0..spectrum.band_count() {
            let intensity = spectrum.disp[band].clamp(0.0, 1.0);
            let cx = MARGIN_X + (band as f32 + 0.5) * p.slot_w;
            let base_y = p.bottom;
            let burst_y = p.bottom - (0.16 + intensity * 0.74) * p.height;
            let tint = palette_tint(palette, band);
            emit_firework(out, band, cx, base_y, burst_y, intensity, tint);
        }
    } else {
        let mid = (p.top + p.bottom) * 0.5;
        let half_h = p.height * 0.5 - 2.0;
        for band in 0..spectrum.band_count() {
            let cx = MARGIN_X + (band as f32 + 0.5) * p.slot_w;
            let inbound = spectrum.disp[band * 2].clamp(0.0, 1.0);
            let outbound = spectrum.disp[band * 2 + 1].clamp(0.0, 1.0);

            emit_firework(
                out,
                band * 2,
                cx,
                mid,
                mid - (0.10 + inbound * 0.82) * half_h,
                inbound,
                palette_tint(palette, band * 2),
            );
            emit_firework(
                out,
                band * 2 + 1,
                cx,
                mid,
                mid + (0.10 + outbound * 0.82) * half_h,
                outbound,
                palette_tint(palette, band * 2 + 1),
            );
        }
    }
}

fn build_radar_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Radar draws static range rings, a rotating sweep, and traffic blips placed
    // around the circle by band index.
    let cx = w * 0.5;
    let cy = p.top + p.height * 0.52;
    let radius = (p.height.min(w - 2.0 * MARGIN_X)) * 0.43;
    let sweep = time * 1.7;

    for ring in 1..=3 {
        let r = radius * ring as f32 / 3.0;
        for i in 0..96 {
            let a = std::f32::consts::TAU * i as f32 / 96.0;
            push_particle(
                out,
                p,
                cx + a.cos() * r,
                cy + a.sin() * r,
                2.2,
                [0.08, 0.95, 0.72, 0.10],
                0.2,
            );
        }
    }

    for i in 0..42 {
        let k = i as f32 / 42.0;
        let r = radius * k;
        push_particle(
            out,
            p,
            cx + sweep.cos() * r,
            cy + sweep.sin() * r,
            3.0 + k * 3.5,
            [0.12, 1.0, 0.78, 0.24 * (1.0 - k * 0.35)],
            0.5,
        );
    }

    for band in 0..spectrum.band_count() {
        let (inbound, outbound) = band_signed_levels(spectrum, band);
        let level = inbound.max(outbound);
        if level <= 0.002 {
            continue;
        }
        let base_angle = std::f32::consts::TAU * band as f32 / spectrum.band_count().max(1) as f32
            - std::f32::consts::FRAC_PI_2;
        let pulse = (time * (0.8 + level * 2.4) + hash01(band as u32) * 2.0).fract();
        let blip_r = radius * (0.18 + level * 0.76);
        let tint = palette_tint(palette, band);
        let alpha = 0.35 + level * 0.65;
        push_bloom_particle(
            out,
            p,
            cx + base_angle.cos() * blip_r,
            cy + base_angle.sin() * blip_r,
            8.0 + level * 28.0,
            firework_color(level, band as u32, tint, alpha),
            level,
        );
        push_bloom_particle(
            out,
            p,
            cx + base_angle.cos() * blip_r,
            cy + base_angle.sin() * blip_r,
            16.0 + pulse * 42.0 * level,
            firework_color(level, band as u32 ^ 0xbeef, tint, (1.0 - pulse) * 0.28),
            level,
        );
        if spectrum.mirrored && outbound > 0.002 {
            let out_angle = base_angle + std::f32::consts::PI;
            push_bloom_particle(
                out,
                p,
                cx + out_angle.cos() * blip_r,
                cy + out_angle.sin() * blip_r,
                8.0 + outbound * 24.0,
                firework_color(
                    outbound,
                    band as u32 ^ 0x7711,
                    palette_tint(palette, band + 1),
                    alpha,
                ),
                outbound,
            );
        }
    }
}

fn build_matrix_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Matrix mode maps each band to a column of falling particles. Higher
    // traffic increases density, size, and brightness.
    for band in 0..spectrum.band_count() {
        let level = band_level(spectrum, band);
        if level <= 0.002 {
            continue;
        }
        let drops = (6.0 + level * 26.0).round() as usize;
        let x0 = MARGIN_X + band as f32 * p.slot_w;
        let tint = palette_tint(palette, band);
        for i in 0..drops {
            let seed = (band as u32 + 1).wrapping_mul(131).wrapping_add(i as u32);
            let speed = 0.16 + level * 0.70 + hash01(seed) * 0.35;
            let y = p.top + ((time * speed + hash01(seed ^ 0xaaa1)).fract()) * p.height;
            let x = x0 + (0.12 + hash01(seed ^ 0xbbb2) * 0.76) * p.slot_w;
            let head = i % 5 == 0;
            let alpha = if head { 0.85 } else { 0.22 + level * 0.42 };
            let size = if head {
                8.0 + level * 8.0
            } else {
                4.0 + level * 7.0
            };
            push_trail_bloom_particle(
                out,
                p,
                TrailParticle {
                    cx: x,
                    cy: y,
                    dx: 0.0,
                    dy: p.height * (0.035 + level * 0.035),
                    size,
                    color: firework_color(level, seed, tint, alpha),
                    intensity: level,
                    steps: 4,
                },
            );
        }
    }
}

fn build_oscilloscope_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Oscilloscope mode samples a synthetic waveform within each band slot.
    // Mirrored outbound traffic flips the wave direction.
    for band in 0..spectrum.band_count() {
        let (inbound, outbound) = band_signed_levels(spectrum, band);
        let level = inbound.max(outbound);
        if level <= 0.002 {
            continue;
        }
        let samples = 28;
        let x0 = MARGIN_X + band as f32 * p.slot_w + p.slot_w * 0.08;
        let center = p.top + p.height * (0.50 + (hash01(band as u32) - 0.5) * 0.16);
        let amp = p.height * (0.025 + level * 0.16);
        let tint = palette_tint(palette, band);
        let direction = if spectrum.mirrored && outbound > inbound {
            -1.0
        } else {
            1.0
        };
        for i in 0..samples {
            let k = i as f32 / (samples - 1) as f32;
            let phase = time * (3.0 + level * 7.0) + k * std::f32::consts::TAU * 2.2;
            let wave = phase.sin() * 0.72 + (phase * 2.3 + band as f32).sin() * 0.28;
            let y = center - wave * amp * direction;
            let x = x0 + k * p.slot_w * 0.84;
            push_trail_bloom_particle(
                out,
                p,
                TrailParticle {
                    cx: x,
                    cy: y,
                    dx: p.slot_w * 0.045,
                    dy: amp * direction * 0.28,
                    size: 4.0 + level * 8.0,
                    color: firework_color(level, band as u32 ^ i as u32, tint, 0.34 + level * 0.62),
                    intensity: level,
                    steps: 3,
                },
            );
        }
        push_flat_rect(
            out,
            p,
            x0,
            center - 0.8,
            x0 + p.slot_w * 0.84,
            center + 0.8,
            [tint[0], tint[1], tint[2], 0.08],
        );
    }
}

fn build_pulse_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Pulse mode emits expanding rings from each active band's vertical level.
    for band in 0..spectrum.band_count() {
        let level = band_level(spectrum, band);
        if level <= 0.002 {
            continue;
        }
        let cx = MARGIN_X + (band as f32 + 0.5) * p.slot_w;
        let cy = p.bottom - (0.20 + level * 0.62) * p.height;
        let tint = palette_tint(palette, band);
        for ring in 0..3 {
            let phase =
                (time * (0.45 + level * 1.6) + ring as f32 * 0.32 + band as f32 * 0.07).fract();
            let radius = p.slot_w * (0.14 + phase * (0.58 + level * 0.42));
            let points = (18.0 + level * 34.0).round() as usize;
            let alpha = (1.0 - phase) * (0.14 + level * 0.48);
            for i in 0..points {
                let a = std::f32::consts::TAU * i as f32 / points as f32;
                push_trail_bloom_particle(
                    out,
                    p,
                    TrailParticle {
                        cx: cx + a.cos() * radius,
                        cy: cy + a.sin() * radius,
                        dx: a.sin() * radius * 0.12,
                        dy: -a.cos() * radius * 0.12,
                        size: 3.8 + level * 8.0,
                        color: firework_color(level, band as u32 ^ (i as u32 * 17), tint, alpha),
                        intensity: level,
                        steps: 3,
                    },
                );
            }
        }
    }
}

fn build_galaxy_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Galaxy mode treats each band as an orbit system whose particle count and
    // orbital radius grow with traffic intensity.
    for band in 0..spectrum.band_count() {
        let level = band_level(spectrum, band);
        if level <= 0.002 {
            continue;
        }
        let cx = MARGIN_X + (band as f32 + 0.5) * p.slot_w;
        let cy = p.top + p.height * (0.24 + 0.52 * hash01((band as u32 + 5) * 19));
        let particles = (8.0 + level * 34.0).round() as usize;
        let tint = palette_tint(palette, band);
        for i in 0..particles {
            let seed = (band as u32 + 1)
                .wrapping_mul(4099)
                .wrapping_add(i as u32 * 97);
            let orbit = p.slot_w * (0.10 + hash01(seed) * (0.34 + level * 0.52));
            let angle = time * (0.55 + level * 2.2) * if i % 2 == 0 { 1.0 } else { -1.0 }
                + hash01(seed ^ 0x9090) * std::f32::consts::TAU;
            let squash = 0.42 + hash01(seed ^ 0x1234) * 0.38;
            push_trail_bloom_particle(
                out,
                p,
                TrailParticle {
                    cx: cx + angle.cos() * orbit,
                    cy: cy + angle.sin() * orbit * squash,
                    dx: -angle.sin() * orbit * 0.18,
                    dy: angle.cos() * orbit * squash * 0.18,
                    size: 3.8 + level * 9.0,
                    color: firework_color(level, seed, tint, 0.28 + level * 0.58),
                    intensity: level,
                    steps: 4,
                },
            );
        }
        push_bloom_particle(
            out,
            p,
            cx,
            cy,
            9.0 + level * 26.0,
            firework_color(level, band as u32 ^ 0xfeed, tint, 0.28 + level * 0.48),
            level,
        );
    }
}

fn build_lightning_instances(
    out: &mut Vec<Inst>,
    spectrum: &Spectrum,
    w: f32,
    h: f32,
    time: f32,
    palette: VisualPalette,
) {
    out.clear();
    let p = plot(w, h, spectrum.band_count());
    push_trail_overlay(out, p);
    // Lightning mode builds a jagged bolt per active band, with occasional
    // short branches to make stronger traffic look more explosive.
    for band in 0..spectrum.band_count() {
        let level = band_level(spectrum, band);
        if level <= 0.002 {
            continue;
        }
        let bolts = (1.0 + level * 3.0).round() as usize;
        let tint = palette_tint(palette, band);
        for bolt in 0..bolts {
            let seed_base = (band as u32 + 1)
                .wrapping_mul(6151)
                .wrapping_add(bolt as u32 * 331);
            let x_base = MARGIN_X + (band as f32 + 0.5) * p.slot_w;
            let segments = (8.0 + level * 18.0).round() as usize;
            let mut prev_x = x_base + (hash01(seed_base) - 0.5) * p.slot_w * 0.45;
            let mut prev_y = p.top + p.height * 0.10;
            for i in 1..=segments {
                let k = i as f32 / segments as f32;
                let jitter = ((time * 9.0 + i as f32 * 1.7 + band as f32).sin()
                    + hash01(seed_base ^ i as u32)
                    - 0.5)
                    * p.slot_w
                    * (0.08 + level * 0.22);
                let x = x_base + jitter;
                let y = p.top + p.height * (0.10 + k * (0.78 * level + 0.12));
                let alpha = 0.22 + level * 0.78;
                push_bloom_particle(
                    out,
                    p,
                    x,
                    y,
                    5.0 + level * 10.0,
                    firework_color(level, seed_base ^ i as u32, tint, alpha),
                    level,
                );
                let x_mid = (prev_x + x) * 0.5;
                let y_mid = (prev_y + y) * 0.5;
                push_flat_rect(
                    out,
                    p,
                    x_mid - 1.4 - level * 2.2,
                    y_mid - (y - prev_y).abs() * 0.50,
                    x_mid + 1.4 + level * 2.2,
                    y_mid + (y - prev_y).abs() * 0.50,
                    [tint[0], tint[1], tint[2], 0.10 + level * 0.24],
                );
                if i % 4 == 0 {
                    let branch_dir = if hash01(seed_base ^ (i as u32 * 11)) > 0.5 {
                        1.0
                    } else {
                        -1.0
                    };
                    for b in 0..4 {
                        let bk = b as f32 / 4.0;
                        push_bloom_particle(
                            out,
                            p,
                            x + branch_dir * p.slot_w * 0.08 * b as f32,
                            y - bk * p.height * 0.06,
                            3.5 + level * 7.0,
                            firework_color(
                                level,
                                seed_base ^ (i as u32 * 23 + b as u32),
                                tint,
                                alpha * (1.0 - bk * 0.7),
                            ),
                            level,
                        );
                    }
                }
                prev_x = x;
                prev_y = y;
            }
        }
    }
}
