use ndarray_images::Image;

use mandybrot::{render_fractal, Complex, Fractal};

const OUTPUT_DIR: &str = "output";
const FILENAME: &str = "grayscale.png";

fn main() {
    let fractal = Fractal::Mandelbrot;

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [2048, 2048];
    let super_samples = 2;
    let data = render_fractal(centre, max_iter, scale, resolution, fractal, super_samples);

    // Convert to normalised f32 values
    let data = data.mapv(|v| v as f32 / max_iter as f32);

    // Create an image from the data
    let filename = format!("{}/{}", OUTPUT_DIR, FILENAME);
    data.save(filename).unwrap();
}
