use enterpolation::{linear::ConstEquidistantLinear, Generator};
use ndarray::{stack, Array2, Array3, Axis};
use ndarray_images::Image;
use palette::LinSrgb;

use mandybrot::{multisample_area, Complex, Fractal};

const OUTPUT_DIR: &str = "output";
const FILENAME: &str = "mandelbrot.png";

const FRACTAL: Fractal<f64> = Fractal::Mandelbrot;

const CENTRE: Complex<f64> = Complex::new(-0.75, 0.0);
const MAX_ITER: u32 = 100;
const SCALE: f64 = 3.0;
const RESOLUTION: [u32; 2] = [1024, 1024];
const SAMPLES: u32 = 4;

fn main() {
    // Generate Mandelbrot data
    let data = multisample_area(CENTRE, MAX_ITER, SCALE, RESOLUTION, FRACTAL, SAMPLES);

    // Convert iteration counts to normalised values (0.0 - 1.0)
    let data = data.mapv(|v| v as f64 / MAX_ITER as f64);

    // Apply the gradient to convert grayscale values to RGB
    let gradient = ConstEquidistantLinear::<f64, _, 3>::equidistant_unchecked([
        LinSrgb::new(0.00, 0.05, 0.20), // Dark blue
        LinSrgb::new(0.70, 0.10, 0.20), // Red tone
        LinSrgb::new(0.95, 0.90, 0.30), // Yellow tone
    ]);
    let coloured_data = data.mapv(|v| gradient.gen(v));

    // Convert from `Array2<LinSrgb<f64>>` to `Array3<f64>`
    let (height, width) = coloured_data.dim();
    let red = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].red);
    let green = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].green);
    let blue = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].blue);

    let data: Array3<f64> = stack![Axis(2), red, green, blue];

    // Save the image
    let filename = format!("{}/{}", OUTPUT_DIR, FILENAME);
    data.save(filename).unwrap();
}
