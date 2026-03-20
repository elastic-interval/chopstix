mod app;
mod camera;
mod constants;
mod gpu;
mod sphere;
mod tensegrity;

use clap::Parser;
use winit::event_loop::EventLoop;

#[derive(Parser)]
#[command(name = "chopstix", about = "GPU tensegrity sphere experiment")]
struct Args {
    /// Sphere frequency (geodesic subdivision level)
    #[arg(short, long, default_value_t = 3)]
    frequency: usize,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    if args.frequency < 1 {
        eprintln!("Frequency must be at least 1");
        std::process::exit(1);
    }

    log::info!("Starting chopstix with frequency {}", args.frequency);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = app::App::new(args.frequency);
    event_loop.run_app(&mut app).expect("Event loop error");
}
