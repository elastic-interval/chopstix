use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

const TITLE_COLOR: Color = Color::rgb(255, 255, 255);
const VALUE_COLOR: Color = Color::rgb(200, 230, 255);
const KEY_COLOR: Color = Color::rgb(140, 200, 255);
const DESC_COLOR: Color = Color::rgb(200, 200, 190);
const INFO_COLOR: Color = Color::rgb(160, 160, 150);

fn mono(color: Color) -> Attrs<'static> {
    Attrs::new().family(Family::Monospace).color(color)
}

pub struct Hud {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    title_buffer: Buffer,
    legend_buffer: Buffer,
    info_buffer: Buffer,
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

        Self {
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            title_buffer,
            legend_buffer,
            info_buffer,
            screen_height: 600.0,
        }
    }

    /// Top-left: mode name and current value
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

    /// Bottom-left: available keys in current mode
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

    /// Bottom-right area: stats
    pub fn set_info(&mut self, text: &str) {
        self.info_buffer.set_text(
            &mut self.font_system,
            text,
            &mono(INFO_COLOR),
            Shaping::Basic,
        );
        self.info_buffer.shape_until_scroll(&mut self.font_system, false);
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

        let text_areas = [
            // Title: top-left
            TextArea {
                buffer: &self.title_buffer,
                left: 16.0,
                top: 12.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: TITLE_COLOR,
                custom_glyphs: &[],
            },
            // Legend: bottom-left
            TextArea {
                buffer: &self.legend_buffer,
                left: 16.0,
                top: legend_top,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: width as i32, bottom: height as i32 },
                default_color: DESC_COLOR,
                custom_glyphs: &[],
            },
            // Info: bottom-right
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
