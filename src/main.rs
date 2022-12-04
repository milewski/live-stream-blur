// #![allow(dead_code, unused)]

use std::borrow::Cow;
use std::time::{Duration, Instant};

use glutin_window::GlutinWindow;
use graphics::{image as create_image};
use image::{DynamicImage, GenericImageView, Pixel};
use image::imageops::{blur, FilterType};
use nokhwa::{Camera, CameraFormat, FrameFormat};
use opengl_graphics::{GlGraphics, OpenGL, Texture, TextureSettings};
use piston::RenderEvent;
use piston::event_loop::{Events, EventSettings};
use piston::window::WindowSettings;
use rustface::{Detector, ImageData};

#[derive(PartialEq)]
struct Resolution {
    width: u32,
    height: u32,
}

struct Settings {
    framerate: u32,
    capture: Resolution,
    detection: Resolution,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            capture: Resolution { width: 1280, height: 720 },
            detection: Resolution { width: 1280, height: 720 },
            // detection: Resolution { width: 640 - 200, height: 480 - 200 },
            framerate: 30,
        }
    }
}

fn main() {
    let options: Settings = Settings {
        ..Settings::default()
    };

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

    let format = CameraFormat::new_from(options.capture.width, options.capture.height, FrameFormat::MJPEG, options.framerate);
    let mut camera = Camera::new(1, Some(format))
        .unwrap();

    camera.open_stream().unwrap();

    let opengl = OpenGL::V4_5;

    let mut window: GlutinWindow = WindowSettings::new("capture", [options.capture.width, options.capture.height])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut gl = GlGraphics::new(opengl);
    let mut events = Events::new(EventSettings::new());

    while let Some(event) = events.next(&mut window) {
        if let Some(args) = event.render_args() {
            let buffer = camera.frame_raw().unwrap();
            let texture = get_image_from_frame(&mut *detector, buffer, &options);

            gl.draw(args.viewport(), |c, g| {
                create_image(&texture, c.transform, g);
            });
        }
    }
}

fn get_millis(duration: Duration) -> u64 {
    duration.as_secs() * 1000u64 + u64::from(duration.subsec_nanos() / 1_000_000)
}

fn detect_face(detector: &mut dyn Detector, options: &Settings) {

}

fn get_image_from_frame(detector: &mut dyn Detector, buffer: Cow<[u8]>, options: &Settings) -> Texture {
    let now = Instant::now();

    let image: DynamicImage = image::load_from_memory(&buffer).unwrap();
    let lower_image: DynamicImage = image
        .resize_exact(options.detection.width, options.detection.height, FilterType::Nearest);

    let transparent_image = DynamicImage::new_rgba8(options.detection.width, options.detection.height);
    let mut transparent_rgba = transparent_image.to_rgba8();
    let mut rgba = image.to_rgba8();

    let lower_luma = lower_image.to_luma8();
    let mut image_data = ImageData::new(&lower_luma, options.detection.width, options.detection.height);
    let faces = detector.detect(&mut image_data);

    for face in faces {
        let bbox = face.bbox();
        let box_x = bbox.x() as u32;
        let box_y = bbox.y() as u32;

        let cropped = lower_image.view(box_x, box_y, bbox.width(), bbox.height()).to_image();
        let blurred = blur(&cropped, 5.0);

        for (x, y, pixel) in blurred.enumerate_pixels() {
            let offset_x = x + box_x;
            let offset_y = y + box_y;
            transparent_rgba.put_pixel(offset_x, offset_y, pixel.to_rgba())
        }
    }

    let resized_image = DynamicImage::from(transparent_rgba)
        .resize_exact(options.capture.width, options.capture.height, FilterType::Nearest)
        .to_rgba8();

    for (x, y, pixel) in resized_image.enumerate_pixels() {
        match pixel.0 {
            [0, 0, 0, 0] => (),
            _ => {
                rgba.put_pixel(x, y, pixel.to_rgba())
            }
        }
    }

    let settings = TextureSettings::new();
    let texture = Texture::from_image(&rgba, &settings);

    println!("Time {} ms", get_millis(now.elapsed()));

    return texture;
}
