// #![allow(dead_code, unused)]

use std::borrow::Cow;
use std::time::{Duration, Instant};

use glutin_window::GlutinWindow;
use graphics::image as create_image;
use image::{DynamicImage, GenericImageView, Pixel};
use image::imageops::blur;
use nokhwa::{Camera, CameraFormat, FrameFormat};
use opengl_graphics::{GlGraphics, OpenGL, Texture, TextureSettings};
use piston::event_loop::{Events, EventSettings};
use piston::RenderEvent;
use piston::window::WindowSettings;
use rustface::{Detector, ImageData};

static WIDTH: u32 = 640;
static HEIGHT: u32 = 480;

fn main() {
    let mut detector = match rustface::create_detector("./model/seeta_fd_frontal_v1.0.bin") {
        Ok(detector) => detector,
        Err(error) => {
            println!("Failed to create detector: {}", error.to_string());
            std::process::exit(1)
        }
    };

    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.7);
    detector.set_slide_window_step(5, 5);

    let mut camera = Camera::new(1, Some(CameraFormat::new_from(WIDTH, HEIGHT, FrameFormat::MJPEG, 30)))
        .unwrap();

    camera.open_stream().unwrap();

    let opengl = OpenGL::V3_2;

    let mut window: GlutinWindow = WindowSettings::new("capture", [WIDTH, HEIGHT])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut gl = GlGraphics::new(opengl);
    let mut events = Events::new(EventSettings::new());

    while let Some(event) = events.next(&mut window) {
        if let Some(args) = event.render_args() {
            let buffer = camera.frame_raw().unwrap();
            let texture = get_image_from_frame(&mut *detector, buffer);

            gl.draw(args.viewport(), |c, g| {
                create_image(&texture, c.transform, g);
            });
        }
    }
}

fn get_millis(duration: Duration) -> u64 {
    duration.as_secs() * 1000u64 + u64::from(duration.subsec_nanos() / 1_000_000)
}

fn get_image_from_frame(detector: &mut dyn Detector, buffer: Cow<[u8]>) -> Texture {
    let now = Instant::now();
    let image: DynamicImage = image::load_from_memory(&buffer).unwrap();

    let mut rgba = image.to_rgba8();
    let luma = image.to_luma8();

    let mut image_data = ImageData::new(&luma, luma.width(), luma.height());
    let faces = detector.detect(&mut image_data);

    for face in faces {
        let bbox = face.bbox();
        let box_x = bbox.x() as u32;
        let box_y = bbox.y() as u32;

        let cropped = rgba.view(box_x, box_y, bbox.width(), bbox.height()).to_image();
        let blurred = blur(&cropped, 5.0);

        for (x, y, pixel) in blurred.enumerate_pixels() {
            let offset_x = x + box_x;
            let offset_y = y + box_y;
            rgba.put_pixel(offset_x, offset_y, pixel.to_rgba())
        }
    }

    let settings = TextureSettings::new();
    let texture = Texture::from_image(&rgba, &settings);

    println!("Time {} ms", get_millis(now.elapsed()));

    return texture;
}
