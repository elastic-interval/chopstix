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

fn mono(color: Color) -> Attrs<'static> {
    Attrs::new().family(Family::Monospace).color(color)
}

const SLIDER_STEPS: usize = 30;

pub struct Hud {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    title_buffer: Buffer,
    legend_buffer: Buffer,
    info_buffer: Buffer,
    slider_buffer: Buffer,
    screen_height: f32,
    show_slider: bool,
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

        let mut slider_buffer = Buffer::new(&mut font_system, Metrics::new(28.0, 32.0));
        slider_buffer.set_size(&mut font_system, Some(200.0), Some(1200.0));

        Self {
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            title_buffer,
            legend_buffer,
            info_buffer,
            slider_buffer,
            screen_height: 600.0,
            show_slider: false,
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

    /// Show a vertical slider. `t` is 0.0..1.0 position, `label` shows the value.
    pub fn set_slider(&mut self, t: f32, label: &str) {
        self.show_slider = true;
        let t = t.clamp(0.0, 1.0);
        let filled = (t * SLIDER_STEPS as f32).round() as usize;

        // Build vertical slider: top = max, bottom = min
        // Each line is one step of the track
        let mut spans: Vec<(String, Attrs)> = Vec::new();
        for i in (0..SLIDER_STEPS).rev() {
            if i < SLIDER_STEPS - 1 {
                spans.push(("\n".to_string(), mono(SLIDER_TRACK)));
            }
            if i == filled {
                spans.push((" \u{25C0} ".to_string(), mono(SLIDER_FILL))); // ◀ marker
            } else if i <= filled {
                spans.push((" \u{2503} ".to_string(), mono(SLIDER_FILL))); // ┃ filled
            } else {
                spans.push((" \u{2502} ".to_string(), mono(SLIDER_TRACK))); // │ empty
            }
        }
        // Label at bottom
        spans.push(("\n".to_string(), mono(SLIDER_LABEL)));
        spans.push((label.to_string(), mono(SLIDER_LABEL)));

        let borrowed_spans: Vec<(&str, Attrs)> = spans.iter().map(|(s, a)| (s.as_str(), a.clone())).collect();
        self.slider_buffer.set_rich_text(
            &mut self.font_system,
            borrowed_spans,
            &mono(SLIDER_TRACK),
            Shaping::Basic,
            None,
        );
        self.slider_buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn hide_slider(&mut self) {
        self.show_slider = false;
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) {
        self.screen_height = height as f32;
        self.viewport.update(queue, Resolution { width, height });

        let legend_lines = self.legend_buffer.lines.len() as f32;
        let legend_height = legend_lines * 26.0;
        let legend_top = self.screen_height - legend_height - 16.0;

        let info_lines = self.info_buffer.lines.len() as f32;
        let info_height = info_lines * 24.0;
        let info_top = self.screen_height - info_height - 16.0;

        // Slider centered vertically on left edge
        let slider_lines = self.slider_buffer.lines.len() as f32;
        let slider_height = slider_lines * 32.0;
        let slider_top = (self.screen_height - slider_height) * 0.5;

        let mut text_areas: Vec<TextArea> = vec![
            TextArea {
                buffer: &self.title_buffer,
                left: 16.0,
                top: 12.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: TITLE_COLOR,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.legend_buffer,
                left: 16.0,
                top: legend_top,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: DESC_COLOR,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.info_buffer,
                left: width as f32 - 350.0,
                top: info_top,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: INFO_COLOR,
                custom_glyphs: &[],
            },
        ];

        if self.show_slider {
            text_areas.push(TextArea {
                buffer: &self.slider_buffer,
                left: 6.0,
                top: slider_top,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: SLIDER_TRACK,
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
