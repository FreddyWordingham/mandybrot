use enterpolation::{linear::ConstEquidistantLinear, Generator};
use ndarray::{stack, Array2, Array3, Axis};
use ndarray_images::Image;
use palette::LinSrgb;

use mandybrot::{sample_area, Complex};

const OUTPUT_DIR: &str = "output";
const FILENAME: &str = "colour.png";

fn main() {
    let gradient = ConstEquidistantLinear::<f32, _, 3>::equidistant_unchecked([
        LinSrgb::new(0.00, 0.05, 0.20), // Dark blue
        LinSrgb::new(0.70, 0.10, 0.20), // Red tone
        LinSrgb::new(0.95, 0.90, 0.30), // Yellow tone
    ]);

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [2048, 2048];

    // Generate Mandelbrot data
    let data = sample_area(centre, max_iter, scale, resolution);

    // Convert iteration counts to normalised values (0.0 - 1.0)
    let data_f32 = data.mapv(|v| v as f32 / max_iter as f32);

    // Apply the gradient to convert greyscale values to RGB
    let coloured_data = data_f32.mapv(|v| gradient.gen(v));

    // Convert from `Array2<LinSrgb<f32>>` to `Array3<f32>`
    let (height, width) = coloured_data.dim();
    let red = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].red);
    let green = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].green);
    let blue = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].blue);

    let data: Array3<f32> = stack![Axis(2), red, green, blue];

    // Save the image
    let filename = format!("{}/{}", OUTPUT_DIR, FILENAME);
    data.save(filename).unwrap();
}
