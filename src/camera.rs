use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub look_at: Vec3,
    width: f32,
    height: f32,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

impl Camera {
    pub fn new(width: f32, height: f32, initial_distance: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, initial_distance),
            look_at: Vec3::ZERO,
            width,
            height,
            dragging: false,
            last_mouse: None,
        }
    }

    pub fn set_size(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    pub fn set_distance(&mut self, distance: f32) {
        let gaze = (self.look_at - self.position).normalize();
        self.position = self.look_at - gaze * distance;
    }

    pub fn mouse_pressed(&mut self, x: f64, y: f64) {
        self.dragging = true;
        self.last_mouse = Some((x, y));
    }

    pub fn mouse_released(&mut self) {
        self.dragging = false;
        self.last_mouse = None;
    }

    pub fn mouse_moved(&mut self, x: f64, y: f64) {
        if !self.dragging {
            return;
        }
        if let Some((lx, ly)) = self.last_mouse {
            let dx = (x - lx) as f32;
            let dy = (y - ly) as f32;
            if dx != 0.0 || dy != 0.0 {
                if let Some(rotation) = self.rotation(dx, dy) {
                    self.position =
                        self.look_at - rotation.transform_vector3(self.look_at - self.position);
                }
            }
        }
        self.last_mouse = Some((x, y));
    }

    pub fn track_target(&mut self, target: Vec3) {
        let offset = target - self.look_at;
        self.look_at += offset;
        self.position += offset;
    }

    pub fn scroll(&mut self, delta: f32) {
        let gaze = self.look_at - self.position;
        let distance = gaze.length();
        let move_amount = delta * distance * 0.1;
        if distance - move_amount > 0.1 {
            self.position += gaze.normalize() * move_amount;
        }
    }

    pub fn mvp_matrix(&self) -> Mat4 {
        let aspect = self.width / self.height;
        let projection =
            Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 1000.0);
        let view = Mat4::look_at_rh(self.position, self.look_at, Vec3::Y);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }

    fn rotation(&self, dx: f32, dy: f32) -> Option<Mat4> {
        let sensitivity = 0.5;
        let dx = dx * sensitivity;
        let dy = dy * sensitivity;

        let up = Vec3::Y;
        let gaze = self.look_at - self.position;
        let right = gaze.cross(up).normalize();
        let dot = gaze.normalize().dot(up);

        let yaw = Quat::from_axis_angle(up, -dx / 100.0);
        let pitch =
            if (dot > 0.0 && dy < 0.0 && dot > 0.95) || (dot < 0.0 && dy > 0.0 && dot < -0.95) {
                Quat::from_axis_angle(right, 0.0)
            } else {
                Quat::from_axis_angle(right, -dy / 100.0)
            };

        let rotation = yaw * pitch;
        Some(Mat4::from_quat(rotation))
    }
}

const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
]);
