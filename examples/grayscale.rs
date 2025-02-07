use enterpolation::{linear::ConstEquidistantLinear, Generator};
use ndarray_images::Image;
use palette::LinSrgb;

use mandybrot::{sample_area, Complex};

fn main() {
    let gradient = ConstEquidistantLinear::<f32, _, 3>::equidistant_unchecked([
        LinSrgb::new(0.00, 0.05, 0.20),
        LinSrgb::new(0.70, 0.10, 0.20),
        LinSrgb::new(0.95, 0.90, 0.30),
    ]);

    let x = 0.55;
    let color = gradient.gen(x);
    println!("{} {} {}", color.red, color.green, color.blue);

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [2048, 2048];
    let data = sample_area(centre, max_iter, scale, resolution);

    // Convert to normalised f32 values
    let data = data.mapv(|v| v as f32 / max_iter as f32);

    // Create an image from the data
    data.save("mandelbrot.png").unwrap();
}
