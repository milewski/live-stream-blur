// #![allow(dead_code, unused)]

use std::borrow::Cow;
use std::time::{Duration, Instant};

use glutin_window::GlutinWindow;
use graphics::{image as create_image};
use image::{DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage};
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
    blur_intensity: f32,
    capture: Resolution,
    detection: Resolution,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            capture: Resolution { width: 1280, height: 720 },
            detection: Resolution { width: 640 - 300, height: 480 - 300 },
            blur_intensity: 1.5,
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

fn loop_faces(
    detector: &mut dyn Detector,
    options: &Settings,
    source: &DynamicImage,
    vase: &DynamicImage,
    callback: fn(&mut RgbaImage, u32, u32, Rgba<u8>) -> (),
) -> RgbaImage {
    let mut output = vase.to_rgba8();
    let luma = source.to_luma8();
    let mut image_data = ImageData::new(&luma, options.detection.width, options.detection.height);
    let faces = detector.detect(&mut image_data);

    for face in faces {
        let bbox = face.bbox();
        let box_x = bbox.x() as u32;
        let box_y = bbox.y() as u32;

        let cropped = source.view(box_x, box_y, bbox.width(), bbox.height()).to_image();
        let blurred = blur(&cropped, options.blur_intensity);

        for (x, y, pixel) in blurred.enumerate_pixels() {
            callback(&mut output, x + box_x, y + box_y, pixel.to_rgba());
        }
    }

    output
}

fn process(
    detector: &mut dyn Detector,
    options: &Settings,
    source: DynamicImage,
) -> RgbaImage {
    let low_resolution_image: DynamicImage = source
        .resize_exact(options.detection.width, options.detection.height, FilterType::Nearest);

    let blank_image = DynamicImage::new_rgba8(options.detection.width, options.detection.height);
    let output_temp = loop_faces(detector, &options, &low_resolution_image, &blank_image, |output, x, y, pixel: Rgba<u8>| {
        output.put_pixel(x, y, pixel);
    });

    let resized_image = DynamicImage::from(output_temp)
        .resize_exact(options.capture.width, options.capture.height, FilterType::Nearest)
        .to_rgba8();

    let mut output = source.to_rgba8();

    for (x, y, pixel) in resized_image.enumerate_pixels() {
        match pixel.0 {
            [0, 0, 0, 0] => (),
            _ => {
                output.put_pixel(x, y, pixel.to_rgba())
            }
        }
    }

    output
}

fn process_light(
    detector: &mut dyn Detector,
    options: &Settings,
    source: DynamicImage,
) -> RgbaImage {
    loop_faces(detector, &options, &source, &source, |output, x, y, pixel: Rgba<u8>| {
        output.put_pixel(x, y, pixel);
    })
}

fn get_image_from_frame(detector: &mut dyn Detector, buffer: Cow<[u8]>, options: &Settings) -> Texture {
    let now = Instant::now();
    let image: DynamicImage = image::load_from_memory(&buffer).unwrap();

    // When capture and detection has the same resolution we can save some work
    // by avoiding lowering the resolution of the source
    let output = match options.detection == options.capture {
        true => process_light(detector, options, image),
        false => process(detector, options, image),
    };

    let settings = TextureSettings::new();
    let texture = Texture::from_image(&output, &settings);

    println!("Time {} ms", get_millis(now.elapsed()));

    texture
}
