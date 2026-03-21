use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

const TITLE_COLOR: Color = Color::rgb(255, 255, 255);
const VALUE_COLOR: Color = Color::rgb(200, 230, 255);
const KEY_COLOR: Color = Color::rgb(140, 200, 255);
const DESC_COLOR: Color = Color::rgb(200, 200, 190);
const INFO_COLOR: Color = Color::rgb(160, 160, 150);
const SLIDER_TRACK: Color = Color::rgb(60, 60, 55);
const SLIDER_FILL: Color = Color::rgb(140, 200, 255);
const SLIDER_LABEL: Color = Color::rgb(200, 220, 255);
const PRETENSION_FILL: Color = Color::rgb(200, 180, 100);
const PRETENSION_LABEL: Color = Color::rgb(230, 210, 140);
const FREQ_NORMAL: Color = Color::rgb(100, 100, 90);
const FREQ_ACTIVE: Color = Color::rgb(255, 255, 255);
const FREQ_HOVER: Color = Color::rgb(180, 200, 255);

fn mono(color: Color) -> Attrs<'static> {
    Attrs::new().family(Family::Monospace).color(color)
}

const SLIDER_STEPS: usize = 30;

pub const FREQ_CHOICES: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
];

pub const FREQ_BUTTON_WIDTH: f32 = 52.0;
pub const FREQ_BAR_HEIGHT: f32 = 36.0;
pub const FREQ_BAR_RIGHT_MARGIN: f32 = 16.0;
pub const FREQ_BAR_TOP: f32 = 12.0;

/// Klein grid: 3 rows × 10 columns of (width, height) pairs.
/// Width must be even, height must be odd. Max 300 in either dimension.
/// Each row holds a constant product (≈ joint count), varying aspect ratio
/// from tall/narrow (left) to wide/short (right).
pub const KLEIN_GRID: &[&[(usize, usize)]] = &[
    // Row 0: product ≈ 2000 (~1000 joints) — narrow to wide
    &[(4, 299), (6, 251), (8, 201), (10, 163), (14, 127), (20, 101), (28, 71), (40, 51), (56, 37), (76, 27)],
    // Row 1: product ≈ 8000 (~4000 joints)
    &[(6, 299), (10, 251), (16, 201), (24, 163), (32, 127), (44, 101), (60, 77), (82, 55), (110, 41), (148, 31)],
    // Row 2: product ≈ 24000 (~12000 joints)
    &[(14, 299), (22, 253), (34, 201), (48, 163), (66, 127), (82, 101), (108, 77), (146, 57), (200, 41), (296, 29)],
];
pub const KLEIN_GRID_COLS: usize = 10;
pub const KLEIN_GRID_ROWS: usize = 3;
pub const KLEIN_BUTTON_WIDTH: f32 = 100.0;
pub const KLEIN_ROW_HEIGHT: f32 = 30.0;
pub const KLEIN_GRID_TOP: f32 = 54.0;
pub const KLEIN_GRID_RIGHT_MARGIN: f32 = 16.0;

/// X offset for each slider column
const SLIDER_COL_1: f32 = 10.0;
const SLIDER_COL_2: f32 = 70.0;
/// Total width of the slider hit zone
const SLIDER_COL_WIDTH: f32 = 55.0;

pub struct Hud {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    title_buffer: Buffer,
    legend_buffer: Buffer,
    info_buffer: Buffer,
    stiffness_slider: Buffer,
    pretension_slider: Buffer,
    freq_buffer: Buffer,
    klein_grid_buffers: Vec<Buffer>,
    screen_width: f32,
    screen_height: f32,
}

impl Hud {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &cache, surface_format);
        let viewport = Viewport::new(device, &cache);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let mut title_buffer = Buffer::new(&mut font_system, Metrics::new(24.0, 30.0));
        title_buffer.set_size(&mut font_system, Some(800.0), Some(40.0));

        let mut legend_buffer = Buffer::new(&mut font_system, Metrics::new(20.0, 26.0));
        legend_buffer.set_size(&mut font_system, Some(600.0), Some(300.0));

        let mut info_buffer = Buffer::new(&mut font_system, Metrics::new(18.0, 24.0));
        info_buffer.set_size(&mut font_system, Some(600.0), Some(100.0));

        let slider_metrics = Metrics::new(22.0, 26.0);
        let mut stiffness_slider = Buffer::new(&mut font_system, slider_metrics);
        stiffness_slider.set_size(&mut font_system, Some(SLIDER_COL_WIDTH), Some(1000.0));
        let mut pretension_slider = Buffer::new(&mut font_system, slider_metrics);
        pretension_slider.set_size(&mut font_system, Some(SLIDER_COL_WIDTH), Some(1000.0));

        let freq_width = FREQ_CHOICES.len() as f32 * FREQ_BUTTON_WIDTH + 20.0;
        let mut freq_buffer = Buffer::new(&mut font_system, Metrics::new(22.0, FREQ_BAR_HEIGHT));
        freq_buffer.set_size(&mut font_system, Some(freq_width), Some(FREQ_BAR_HEIGHT + 4.0));

        let klein_row_width = KLEIN_GRID_COLS as f32 * KLEIN_BUTTON_WIDTH + 20.0;
        let klein_grid_buffers: Vec<Buffer> = (0..KLEIN_GRID_ROWS)
            .map(|_| {
                let mut buf = Buffer::new(&mut font_system, Metrics::new(18.0, KLEIN_ROW_HEIGHT));
                buf.set_size(&mut font_system, Some(klein_row_width), Some(KLEIN_ROW_HEIGHT + 4.0));
                buf
            })
            .collect();

        Self {
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            title_buffer,
            legend_buffer,
            info_buffer,
            stiffness_slider,
            pretension_slider,
            freq_buffer,
            klein_grid_buffers,
            screen_width: 1280.0,
            screen_height: 600.0,
        }
    }

    pub fn set_title(&mut self, mode: &str, value: &str) {
        let spans: Vec<(&str, Attrs)> = vec![
            (mode, mono(TITLE_COLOR)),
            ("  ", mono(TITLE_COLOR)),
            (value, mono(VALUE_COLOR)),
        ];
        self.title_buffer.set_rich_text(
            &mut self.font_system,
            spans,
            &mono(TITLE_COLOR),
            Shaping::Basic,
            None,
        );
        self.title_buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn set_legend(&mut self, lines: &[(&str, &str)]) {
        let mut spans: Vec<(&str, Attrs)> = Vec::new();
        for (i, (key, desc)) in lines.iter().enumerate() {
            if i > 0 {
                spans.push(("\n", mono(DESC_COLOR)));
            }
            if key.is_empty() {
                spans.push((desc, mono(INFO_COLOR)));
            } else {
                spans.push((key, mono(KEY_COLOR)));
                spans.push(("  ", mono(DESC_COLOR)));
                spans.push((desc, mono(DESC_COLOR)));
            }
        }
        self.legend_buffer.set_rich_text(
            &mut self.font_system,
            spans,
            &mono(DESC_COLOR),
            Shaping::Basic,
            None,
        );
        self.legend_buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn set_info(&mut self, text: &str) {
        self.info_buffer.set_text(
            &mut self.font_system,
            text,
            &mono(INFO_COLOR),
            Shaping::Basic,
        );
        self.info_buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn set_freq_bar(&mut self, current: usize, hover_index: Option<usize>) {
        let mut spans: Vec<(String, Attrs)> = Vec::new();
        for (i, &freq) in FREQ_CHOICES.iter().enumerate() {
            let label = format!("{:>3} ", freq);
            let color = if freq == current {
                FREQ_ACTIVE
            } else if hover_index == Some(i) {
                FREQ_HOVER
            } else {
                FREQ_NORMAL
            };
            spans.push((label, mono(color)));
        }
        let borrowed: Vec<(&str, Attrs)> = spans.iter().map(|(s, a)| (s.as_str(), a.clone())).collect();
        self.freq_buffer.set_rich_text(
            &mut self.font_system,
            borrowed,
            &mono(FREQ_NORMAL),
            Shaping::Basic,
            None,
        );
        self.freq_buffer.shape_until_scroll(&mut self.font_system, false);
    }

    fn build_slider(buffer: &mut Buffer, font_system: &mut FontSystem, t: f32, label: &str, fill_color: Color, label_color: Color, hovered: bool) {
        let t = t.clamp(0.0, 1.0);
        let filled = (t * SLIDER_STEPS as f32).round() as usize;
        let track_color = if hovered { Color::rgb(100, 100, 90) } else { SLIDER_TRACK };

        let mut spans: Vec<(String, Attrs)> = Vec::new();
        // Title at top
        spans.push((label.to_string(), mono(label_color)));
        spans.push(("\n".to_string(), mono(track_color)));

        for i in (0..SLIDER_STEPS).rev() {
            if i == filled {
                spans.push((" \u{25C0}\n".to_string(), mono(fill_color)));
            } else if i <= filled {
                spans.push((" \u{2503}\n".to_string(), mono(fill_color)));
            } else {
                spans.push((" \u{2502}\n".to_string(), mono(track_color)));
            }
        }

        let borrowed: Vec<(&str, Attrs)> = spans.iter().map(|(s, a)| (s.as_str(), a.clone())).collect();
        buffer.set_rich_text(
            font_system,
            borrowed,
            &mono(SLIDER_TRACK),
            Shaping::Basic,
            None,
        );
        buffer.shape_until_scroll(font_system, false);
    }

    pub fn set_stiffness_slider(&mut self, t: f32, label: &str, hovered: bool) {
        Self::build_slider(&mut self.stiffness_slider, &mut self.font_system, t, label, SLIDER_FILL, SLIDER_LABEL, hovered);
    }

    pub fn set_pretension_slider(&mut self, t: f32, label: &str, hovered: bool) {
        Self::build_slider(&mut self.pretension_slider, &mut self.font_system, t, label, PRETENSION_FILL, PRETENSION_LABEL, hovered);
    }

    pub fn set_klein_grid(
        &mut self,
        current_width: usize,
        current_height: usize,
        hover: Option<(usize, usize)>, // (row, col)
    ) {
        for (row_idx, row) in KLEIN_GRID.iter().enumerate() {
            let mut spans: Vec<(String, Attrs)> = Vec::new();
            for (col_idx, &(w, h)) in row.iter().enumerate() {
                let label = format!("{:>3}x{:<4}", w, h);
                let color = if w == current_width && h == current_height {
                    FREQ_ACTIVE
                } else if hover == Some((row_idx, col_idx)) {
                    FREQ_HOVER
                } else {
                    FREQ_NORMAL
                };
                spans.push((label, mono(color)));
                // padding
                if col_idx < row.len() - 1 {
                    spans.push(("  ".to_string(), mono(FREQ_NORMAL)));
                }
            }
            let borrowed: Vec<(&str, Attrs)> = spans.iter().map(|(s, a)| (s.as_str(), a.clone())).collect();
            self.klein_grid_buffers[row_idx].set_rich_text(
                &mut self.font_system,
                borrowed,
                &mono(FREQ_NORMAL),
                Shaping::Basic,
                None,
            );
            self.klein_grid_buffers[row_idx].shape_until_scroll(&mut self.font_system, false);
        }
    }

    pub fn klein_grid_left(&self) -> f32 {
        self.screen_width - KLEIN_GRID_COLS as f32 * KLEIN_BUTTON_WIDTH - KLEIN_GRID_RIGHT_MARGIN
    }

    pub fn freq_bar_left(&self) -> f32 {
        self.screen_width - FREQ_CHOICES.len() as f32 * FREQ_BUTTON_WIDTH - FREQ_BAR_RIGHT_MARGIN
    }

    /// Which slider column (if any) is the cursor over? Returns 0 for stiffness, 1 for pretension.
    pub fn slider_hit(&self, cursor_x: f64) -> Option<usize> {
        let x = cursor_x as f32;
        if x >= SLIDER_COL_1 && x < SLIDER_COL_1 + SLIDER_COL_WIDTH {
            Some(0) // stiffness
        } else if x >= SLIDER_COL_2 && x < SLIDER_COL_2 + SLIDER_COL_WIDTH {
            Some(1) // pretension
        } else {
            None
        }
    }

    /// Map a cursor Y position to a slider value 0.0..1.0.
    /// Top of slider = 1.0 (max), bottom = 0.0 (min).
    pub fn slider_y_to_t(&self, cursor_y: f64) -> f32 {
        let slider_lines = (SLIDER_STEPS + 2) as f32; // steps + label + gap
        let slider_height = slider_lines * 26.0;
        let slider_top = (self.screen_height - slider_height) * 0.5;
        // The track starts after the label line (1 line = 26px)
        let track_top = slider_top + 26.0;
        let track_height = SLIDER_STEPS as f32 * 26.0;

        let y = cursor_y as f32;
        // Top of track = high value (1.0), bottom = low value (0.0)
        let t = 1.0 - (y - track_top) / track_height;
        t.clamp(0.0, 1.0)
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) {
        self.screen_width = width as f32;
        self.screen_height = height as f32;
        self.viewport.update(queue, Resolution { width, height });

        let legend_lines = self.legend_buffer.lines.len() as f32;
        let legend_height = legend_lines * 26.0;
        let legend_top = self.screen_height - legend_height - 16.0;

        let info_lines = self.info_buffer.lines.len() as f32;
        let info_height = info_lines * 24.0;
        let info_top = self.screen_height - info_height - 16.0;

        let slider_lines = self.stiffness_slider.lines.len() as f32;
        let slider_height = slider_lines * 26.0;
        let slider_top = (self.screen_height - slider_height) * 0.5;

        let freq_left = self.freq_bar_left();
        let bounds = TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 };

        let mut text_areas: Vec<TextArea> = vec![
            TextArea {
                buffer: &self.title_buffer,
                left: 16.0,
                top: 12.0,
                scale: 1.0,
                bounds,
                default_color: TITLE_COLOR,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.legend_buffer,
                left: 16.0,
                top: legend_top,
                scale: 1.0,
                bounds,
                default_color: DESC_COLOR,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.info_buffer,
                left: width as f32 - 350.0,
                top: info_top,
                scale: 1.0,
                bounds,
                default_color: INFO_COLOR,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.freq_buffer,
                left: freq_left,
                top: FREQ_BAR_TOP,
                scale: 1.0,
                bounds,
                default_color: FREQ_NORMAL,
                custom_glyphs: &[],
            },
            // Stiffness slider: left column
            TextArea {
                buffer: &self.stiffness_slider,
                left: SLIDER_COL_1,
                top: slider_top,
                scale: 1.0,
                bounds,
                default_color: SLIDER_TRACK,
                custom_glyphs: &[],
            },
            // Pretension slider: second column
            TextArea {
                buffer: &self.pretension_slider,
                left: SLIDER_COL_2,
                top: slider_top,
                scale: 1.0,
                bounds,
                default_color: SLIDER_TRACK,
                custom_glyphs: &[],
            },
        ];

        // Klein grid rows
        let klein_left = self.klein_grid_left();
        for (row_idx, buf) in self.klein_grid_buffers.iter().enumerate() {
            text_areas.push(TextArea {
                buffer: buf,
                left: klein_left,
                top: KLEIN_GRID_TOP + row_idx as f32 * KLEIN_ROW_HEIGHT,
                scale: 1.0,
                bounds,
                default_color: FREQ_NORMAL,
                custom_glyphs: &[],
            });
        }

        self.text_renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.text_renderer.render(&self.atlas, &self.viewport, render_pass).unwrap();
    }
}
