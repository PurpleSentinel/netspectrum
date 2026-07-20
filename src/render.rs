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
    window::WindowBuilder,
};

use crate::audio::{AudioConfig, AudioControl};
use crate::bands::{human_rate, Mode, Spectrum};
use crate::capture::PacketMeta;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    let mut audio = AudioControl::new(audio_config);
    let mut last = Instant::now();
    let mut label_refresh = 0.0f32;
    let mut instances: Vec<Inst> = Vec::with_capacity(MAX_INSTANCES);

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
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
                match code {
                    KeyCode::Digit1 => spectrum.set_mode(Mode::Protocol),
                    KeyCode::Digit2 => spectrum.set_mode(Mode::Ports),
                    KeyCode::Digit3 => spectrum.set_mode(Mode::Hosts),
                    KeyCode::Digit4 => spectrum.set_mode(Mode::Hybrid),
                    KeyCode::Digit5 => spectrum.set_mode(Mode::Sizes),
                    KeyCode::KeyS => segments = !segments,
                    KeyCode::KeyW => {
                        window_decorated = !window_decorated;
                        window.set_decorations(window_decorated);
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
                let now = Instant::now();
                let dt = (now - last).as_secs_f32();
                last = now;

                for m in rx.try_iter() {
                    spectrum.ingest(&m);
                }
                spectrum.tick(dt);
                audio.update(&spectrum);

                label_refresh -= dt;
                if spectrum.labels_dirty || label_refresh <= 0.0 {
                    rebuild_labels(&mut font_system, &mut label_bufs, &spectrum, &config);
                    spectrum.labels_dirty = false;
                    label_refresh = 0.25;
                }

                let w = config.width as f32;
                let h = config.height as f32;
                build_instances(&mut instances, &spectrum, w, h, segments, transparent_bars);
                queue.write_buffer(&inst_buf, 0, bytemuck::cast_slice(&instances));

                // Header text.
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
                let header = format!(
                    "NETSPECTRUM {} [{}]  in {}  out {}   1-5 modes  S seg  A {}  T {}  W {}  Z {}  X {}  Q quit",
                    iface,
                    spectrum.mode.name(),
                    human_rate(spectrum.rate_in),
                    human_rate(spectrum.rate_out),
                    audio_label,
                    tone_label,
                    frame_label,
                    background_label,
                    bars_label,
                );
                header_buf.set_size(&mut font_system, w, 40.0);
                header_buf.set_text(
                    &mut font_system,
                    &header,
                    Attrs::new().family(Family::Monospace),
                    Shaping::Advanced,
                );

                let mut areas: Vec<TextArea> = vec![TextArea {
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
                }];

                let n = spectrum.band_count().max(1) as f32;
                let slot_w = (w - 2.0 * MARGIN_X) / n;
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

fn window_frame_label(decorated: bool) -> &'static str {
    if decorated {
        "frame"
    } else {
        "bare"
    }
}

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
        background_color, build_instances, preferred_alpha_mode, surface_extent, window_frame_label,
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
                [1.0, 1.0, 0.0, 0.0],
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
                    [v, 0.0, seg_flag, 0.0],
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
                    [p, 1.0, 0.0, 0.0],
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
                [1.0, 1.0, 0.0, 0.0],
            );
            // Centre line tick.
            push(
                out,
                x0,
                mid + 1.0,
                x1,
                mid - 1.0,
                [1.0, 1.0, 1.0, 0.16],
                [0.0, 1.0, 0.0, 0.0],
            );

            let up = spectrum.disp[g * 2];
            if up > 0.002 {
                let tip = mid - up * half_h;
                push(out, x0, mid, x1, tip, tint_in, [up, 0.0, seg_flag, 0.0]);
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
                    [pu, 1.0, 0.0, 0.0],
                );
            }

            let down = spectrum.disp[g * 2 + 1];
            if down > 0.002 {
                let tip = mid + down * half_h;
                push(out, x0, mid, x1, tip, tint_out, [down, 0.0, seg_flag, 0.0]);
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
                    [pd, 1.0, 0.0, 0.0],
                );
            }
        }
    }
}
