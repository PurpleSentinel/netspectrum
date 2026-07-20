// Instanced bar renderer.
// rect   = (x0, y0, x1, y1) in NDC; y0 is the bar base, y1 the lit tip.
// color  = tint (rgb) and base alpha.
// params = (value, draw_mode, segments_flag, unused)
//   value: bar height as fraction of full scale -- drives the colour ramp
//   draw_mode 0.0: draw bar segment
//   draw_mode 1.0: draw solid colour (peak caps, ghost slots)
//   draw_mode 2.0: draw soft particle sprite
//   segments_flag > 0.5: quantise into LED segments

struct VsIn {
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, vin: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    let x = mix(vin.rect.x, vin.rect.z, c.x);
    let y = mix(vin.rect.y, vin.rect.w, c.y);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = c;
    out.color = vin.color;
    out.params = vin.params;
    return out;
}

@fragment
fn fs_main(vout: VsOut) -> @location(0) vec4<f32> {
    let value = vout.params.x;
    let draw_mode = vout.params.y;
    let seg_flag = vout.params.z;

    if (draw_mode > 1.5) {
        let p = vout.uv * 2.0 - vec2<f32>(1.0, 1.0);
        let dist = length(p);
        let glow = smoothstep(1.0, 0.0, dist);
        let core = smoothstep(0.22, 0.0, dist);
        let alpha = vout.color.a * (glow * glow + core * 0.85);
        let col = vout.color.rgb + vec3<f32>(core * 0.45);
        return vec4<f32>(col, alpha);
    }

    if (draw_mode > 0.5) {
        return vout.color;
    }

    // Level at this pixel as a fraction of full scale.
    let level = vout.uv.y * value;

    let low = vec3<f32>(0.10, 0.95, 0.45);   // green
    let mid = vec3<f32>(0.98, 0.86, 0.20);   // amber
    let hot = vec3<f32>(1.00, 0.23, 0.25);   // red

    var col = mix(low, mid, smoothstep(0.45, 0.70, level));
    col = mix(col, hot, smoothstep(0.78, 0.92, level));
    col = col * vout.color.rgb;

    // Glow bloom toward the lit tip of the bar.
    col = col + col * smoothstep(0.72, 1.0, vout.uv.y) * 0.5;

    var alpha = vout.color.a;

    // LED segmentation across full scale.
    if (seg_flag > 0.5) {
        let s = fract(level * 30.0);
        if (s > 0.72) {
            alpha = alpha * 0.10;
        }
    }

    // Soften the vertical edges of each bar.
    let ex = smoothstep(0.0, 0.06, vout.uv.x) * smoothstep(0.0, 0.06, 1.0 - vout.uv.x);
    alpha = alpha * (0.35 + 0.65 * ex);

    return vec4<f32>(col, alpha);
}
