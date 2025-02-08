use enterpolation::{linear::ConstEquidistantLinear, Generator};
use ndarray::{stack, Array2, Array3, Axis};
use ndarray_images::Image;
use palette::{LinSrgb, Srgb};

use mandybrot::{render_attractor, Attractor, Complex};

const OUTPUT_DIR: &str = "output";
const FILENAME: &str = "ikdea.png";

const ATTRACTOR: Attractor<f64> = Attractor::Ikeda { u: 0.918 };

const CENTRE: Complex<f64> = Complex::new(3.0, 2.0);
const MAX_ITER: u32 = 1000;
const SCALE: f64 = 10.0;
const RESOLUTION: [u32; 2] = [1024, 1024];
const COLOURS: [&str; 8] = [
    "#FDDC97", // Gentle yellow
    "#FCA07E", // Soft orange
    "#F76C40", // Vibrant orange
    "#E44E3F", // Red-orange
    "#BA256A", // Magenta
    "#8A197F", // Warm purple
    "#5B0E78", // Rich purple
    "#2C003E", // Deep purple
];

fn main() {
    // Generate Mandelbrot data
    let mut data = render_attractor(CENTRE, CENTRE, MAX_ITER, SCALE, RESOLUTION, ATTRACTOR);

    let n = 40;
    for xi in -n..n {
        for yi in -n..n {
            let x = xi as f64 / n as f64;
            let y = yi as f64 / n as f64;
            let start = Complex::new(x, y);
            data += &render_attractor(start, CENTRE, MAX_ITER, SCALE, RESOLUTION, ATTRACTOR);
        }
    }

    // Convert iteration counts to normalised values (0.0 - 1.0)
    let min = *data.iter().min().unwrap() as f64;
    let max = *data.iter().max().unwrap() as f64;
    let range = (max - min) as f64;
    let data = data.mapv(|v| (v as f64 - min) / range);

    // Apply gamma correction
    let data = data.mapv(|v| v.powf(0.25));

    // Apply the gradient to convert greyscale values to RGB
    let gradient = create_gradient(COLOURS);
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

fn hex_to_lin_srgb(hex: &str) -> LinSrgb<f64> {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).expect("Invalid hex code");
    let g = u8::from_str_radix(&hex[2..4], 16).expect("Invalid hex code");
    let b = u8::from_str_radix(&hex[4..6], 16).expect("Invalid hex code");
    Srgb::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0).into_linear()
}

fn create_gradient<const N: usize>(
    hexes: [&str; N],
) -> ConstEquidistantLinear<f64, LinSrgb<f64>, N> {
    let colors = hexes.map(hex_to_lin_srgb);
    ConstEquidistantLinear::equidistant_unchecked(colors)
}
