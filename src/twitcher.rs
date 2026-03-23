/// Muscle animation system — creates a traveling sine wave of contraction
/// along a sequence of intervals, producing peristaltic locomotion.
///
/// Each muscle's ideal length oscillates sinusoidally with a phase offset
/// based on its position in the sequence, so the contraction wave travels
/// smoothly around the structure.

pub struct Twitcher {
    /// Indices into the elastic interval arrays that are muscles
    muscle_indices: Vec<usize>,
    /// Phase offset for each muscle (radians, evenly spaced over one full wave)
    phase_offsets: Vec<f32>,
    /// Base ideal lengths for all elastic intervals
    base_ideals: Vec<f32>,
    /// Current phase (advances each tick)
    phase: f32,
    /// Phase increment per physics frame
    phase_speed: f32,
    /// Contraction amplitude (0.0 = none, 0.1 = 10% shorter at peak)
    amplitude: f32,
}

impl Twitcher {
    pub fn new(
        muscle_indices: Vec<usize>,
        base_ideals: Vec<f32>,
        phase_speed: f32,
        amplitude: f32,
    ) -> Self {
        let n = muscle_indices.len().max(1) as f32;
        let phase_offsets: Vec<f32> = (0..muscle_indices.len())
            .map(|i| i as f32 / n * std::f32::consts::TAU)
            .collect();
        Self {
            muscle_indices,
            phase_offsets,
            base_ideals,
            phase: 0.0,
            phase_speed,
            amplitude,
        }
    }

    /// Create a twitcher for a Möbius band.
    /// Muscles are the pull-edge intervals in physical order around the band.
    /// Each joint produces two elastic intervals: index 0,2,4,... are pull-edge
    /// (along the band), index 1,3,5,... are pull-width (across).
    /// We use only pull-edge, in the order they appear around the loop,
    /// so the sine wave phase maps directly to physical position.
    pub fn for_mobius(base_ideals: Vec<f32>, joint_count: usize) -> Self {
        // Pull-edge intervals are at elastic indices 0, 2, 4, ...
        // corresponding to joints 0, 1, 2, ... around the band
        let muscles: Vec<usize> = (0..joint_count).map(|i| i * 2).collect();

        Self::new(
            muscles,
            base_ideals,
            0.0016, // ~3927 frames per full cycle (~65 seconds at 60fps)
            0.10,
        )
    }

    /// Advance one physics frame. Always returns true since the wave is continuous.
    pub fn tick(&mut self) -> bool {
        if self.muscle_indices.is_empty() {
            return false;
        }
        self.phase += self.phase_speed;
        if self.phase > std::f32::consts::TAU {
            self.phase -= std::f32::consts::TAU;
        }
        true
    }

    /// Compute current ideal lengths with the sine wave applied to muscles.
    pub fn current_ideals(&self, pull_k_at_1m: f32) -> (Vec<f32>, Vec<f32>) {
        let mut ideals = self.base_ideals.clone();

        for (i, &elastic_idx) in self.muscle_indices.iter().enumerate() {
            if elastic_idx < ideals.len() {
                let wave = (self.phase + self.phase_offsets[i]).sin();
                // Map sin [-1,1] to contraction [0, amplitude]: peak contraction at sin=1
                let contraction = self.amplitude * (1.0 + wave) * 0.5;
                ideals[elastic_idx] = self.base_ideals[elastic_idx] * (1.0 - contraction);
            }
        }

        let ks: Vec<f32> = ideals.iter().map(|&l| pull_k_at_1m / l).collect();
        (ideals, ks)
    }
}
