use crate::gpu::growable::GrowablePhysics;

const DEFAULT_APPROACH_FRAMES: u32 = 60;

pub struct ApproachingInterval {
    pub gpu_index: usize,
    pub is_push: bool,
    pub start_ideal: f32,
    pub target_ideal: f32,
    pub target_k: f32,
    pub remaining_frames: u32,
    pub total_frames: u32,
}

pub struct ApproachManager {
    intervals: Vec<ApproachingInterval>,
}

impl ApproachManager {
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    pub fn add_elastic(
        &mut self,
        gpu_index: usize,
        start_ideal: f32,
        target_ideal: f32,
        target_k: f32,
    ) {
        self.intervals.push(ApproachingInterval {
            gpu_index,
            is_push: false,
            start_ideal,
            target_ideal,
            target_k,
            remaining_frames: DEFAULT_APPROACH_FRAMES,
            total_frames: DEFAULT_APPROACH_FRAMES,
        });
    }

    pub fn add_push(
        &mut self,
        gpu_index: usize,
        start_ideal: f32,
        target_ideal: f32,
        target_k: f32,
    ) {
        self.intervals.push(ApproachingInterval {
            gpu_index,
            is_push: true,
            start_ideal,
            target_ideal,
            target_k,
            remaining_frames: DEFAULT_APPROACH_FRAMES,
            total_frames: DEFAULT_APPROACH_FRAMES,
        });
    }

    /// Advance all approaching intervals by one frame.
    /// Writes updated ideals and K values to GPU buffers.
    /// Returns true if any intervals were updated.
    pub fn tick(&mut self, physics: &GrowablePhysics, queue: &wgpu::Queue) -> bool {
        if self.intervals.is_empty() {
            return false;
        }

        let mut elastic_updates: Vec<(usize, f32, f32)> = Vec::new();
        let mut push_updates: Vec<(usize, f32, f32)> = Vec::new();

        for interval in &mut self.intervals {
            if interval.remaining_frames == 0 {
                continue;
            }
            interval.remaining_frames -= 1;
            let progress = 1.0
                - (interval.remaining_frames as f32 / interval.total_frames as f32);
            let ideal = interval.start_ideal
                + (interval.target_ideal - interval.start_ideal) * progress;
            let k = interval.target_k * (interval.target_ideal / ideal);

            if interval.is_push {
                push_updates.push((interval.gpu_index, ideal, k));
            } else {
                elastic_updates.push((interval.gpu_index, ideal, k));
            }
        }

        // Write updates to GPU
        for (idx, ideal, k) in &elastic_updates {
            physics.write_elastic_ideal_at(queue, *idx, *ideal, *k);
        }
        for (idx, ideal, k) in &push_updates {
            physics.write_push_ideal_at(queue, *idx, *ideal, *k);
        }

        // Remove completed intervals
        self.intervals.retain(|i| i.remaining_frames > 0);

        !elastic_updates.is_empty() || !push_updates.is_empty()
    }

    pub fn all_settled(&self) -> bool {
        self.intervals.is_empty()
    }
}
