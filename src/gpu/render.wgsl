struct Uniforms {
    mvp_matrix: mat4x4<f32>,
};
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Instance attributes
    @location(3) start: vec3<f32>,
    @location(4) radius_factor: f32,
    @location(5) end: vec3<f32>,
    @location(6) material_type: u32,
    @location(7) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) material_type: u32,
};

fn build_cylinder_matrix(start: vec3<f32>, end: vec3<f32>, radius_factor: f32) -> mat4x4<f32> {
    let direction = end - start;
    let length = length(direction);
    let base_radius = 0.04;
    let radius = base_radius * radius_factor;

    if (length < 0.0001) {
        let midpoint = (start + end) * 0.5;
        return mat4x4<f32>(
            vec4<f32>(radius, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.001, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, radius, 0.0),
            vec4<f32>(midpoint.x, midpoint.y, midpoint.z, 1.0)
        );
    }

    let midpoint = (start + end) * 0.5;
    let y_axis = normalize(direction);

    var x_axis: vec3<f32>;
    if (abs(y_axis.y) < 0.999) {
        x_axis = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), y_axis));
    } else {
        x_axis = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), y_axis));
    }

    let z_axis = cross(x_axis, y_axis);

    return mat4x4<f32>(
        vec4<f32>(x_axis * radius, 0.0),
        vec4<f32>(y_axis * length, 0.0),
        vec4<f32>(z_axis * radius, 0.0),
        vec4<f32>(midpoint, 1.0)
    );
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let model_matrix = build_cylinder_matrix(in.start, in.end, in.radius_factor);
    let world_position = model_matrix * vec4<f32>(in.position, 1.0);

    let normal_matrix = mat3x3<f32>(
        normalize(model_matrix[0].xyz),
        normalize(model_matrix[1].xyz),
        normalize(model_matrix[2].xyz)
    );
    let world_normal = normalize(normal_matrix * in.normal);

    var out: VertexOutput;
    out.clip_position = uniforms.mvp_matrix * world_position;
    out.world_position = world_position.xyz;
    out.world_normal = world_normal;
    out.uv = in.uv;
    out.color = in.color;
    out.material_type = in.material_type;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let normal = normalize(in.world_normal);

    let ambient = 0.25;
    let diffuse = max(dot(normal, light_dir), 0.0);

    var base_color: vec3<f32>;
    switch(in.material_type) {
        case 0u: {
            // Push (strut) - metallic silver tint
            base_color = in.color.rgb * vec3<f32>(0.95, 0.97, 1.0);
            break;
        }
        case 1u: {
            // Pull (cable)
            base_color = in.color.rgb;
            break;
        }
        default: {
            base_color = in.color.rgb;
            break;
        }
    }

    let lighting = ambient + diffuse * 0.75;
    let final_color = base_color * lighting;
    let gamma_corrected = pow(final_color, vec3<f32>(1.0/2.2));

    return vec4<f32>(gamma_corrected, in.color.a);
}
