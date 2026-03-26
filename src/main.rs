mod app;
mod build;
mod camera;
mod constants;
mod gpu;
mod klein;
mod mobius;
mod sphere;
mod tensegrity;
mod twitcher;

use winit::event_loop::EventLoop;

use build::fabric_library::FabricName;

#[derive(Clone, Debug)]
pub enum ShapeConfig {
    Sphere { frequency: usize },
    Klein { width: usize, height: usize, shift: usize },
    Mobius { segments: usize },
    Tenscript { fabric: FabricName },
}

fn main() {
    env_logger::init();
    log::info!("Starting chopstix");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = app::App::new(ShapeConfig::Sphere { frequency: 1 });
    event_loop.run_app(&mut app).expect("Event loop error");
}
