use enterpolation::{linear::Linear, Generator};
use ndarray::{stack, Array2, Array3, Axis};
use ndarray_images::Image;
use palette::LinSrgb;
use palette::Srgb;

use mandybrot::{render_attractor, Parameters};

const INPUT_DIR: &str = "input";
const OUTPUT_DIR: &str = "output";
const PARAMETERS_NAME: &str = "parameters.yaml";
const IMAGE_NAME: &str = "attractor.png";

type P = f32;

fn main() {
    let params: Parameters<P> = serde_yaml::from_str(
        &std::fs::read_to_string(format!("{}/{}", INPUT_DIR, PARAMETERS_NAME)).unwrap(),
    )
    .unwrap();

    // Generate Mandelbrot data
    let data = render_attractor(
        params.start,
        params.centre,
        params.max_iter,
        params.scale,
        params.resolution,
        params.attractor,
    );

    // Convert iteration counts to normalised values (0.0 - 1.0)
    let min = *data.iter().min().unwrap() as f64;
    let max = *data.iter().max().unwrap() as f64;
    let range = (max - min) as f64;
    let data = data.mapv(|v| (v as f64 - min) / range);

    // Apply gamma correction
    let data = data.mapv(|v| v.powf(params.gamma as f64));

    // Create a colour gradient
    let hexes: Vec<&str> = params.colours.iter().map(|s| s.as_str()).collect();
    let colours: Vec<_> = hexes.iter().map(|&hex| hex_to_lin_srgb(hex)).collect();
    let num_colours = colours.len();
    let gradient = Linear::builder()
        .elements(colours)
        .knots(linspace(num_colours))
        .build()
        .expect("Failed to build gradient.");

    // Apply the gradient convert greyscale values to RGB
    let coloured_data = data.mapv(|v| gradient.gen(v));

    // Convert from `Array2<LinSrgb<f64>>` to `Array3<f64>`
    let (height, width) = coloured_data.dim();
    let red = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].red);
    let green = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].green);
    let blue = Array2::from_shape_fn((height, width), |(y, x)| coloured_data[(y, x)].blue);

    let data: Array3<f64> = stack![Axis(2), red, green, blue];

    // Save the image
    let filename = format!("{}/{}", OUTPUT_DIR, IMAGE_NAME);
    data.save(filename).unwrap();
}

fn hex_to_lin_srgb(hex: &str) -> LinSrgb<f64> {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).expect("Invalid hex code");
    let g = u8::from_str_radix(&hex[2..4], 16).expect("Invalid hex code");
    let b = u8::from_str_radix(&hex[4..6], 16).expect("Invalid hex code");
    Srgb::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0).into_linear()
}

// fn create_gradient<const N: usize>(
//     hexes: [&str; N],
// ) -> ConstEquidistantLinear<f64, LinSrgb<f64>, N> {
//     let colors = hexes.map(hex_to_lin_srgb);
//     ConstEquidistantLinear::equidistant_unchecked(colors)
// }

// fn create_gradient(hexes: &[&str]) -> enterpolation::linear::Linear<f64, LinSrgb<f64>> {
//     let colors: Vec<_> = hexes.iter().map(|&hex| hex_to_lin_srgb(hex)).collect();
//     ConstEquidistantLinear::equidistant_unchecked(colors)
// }

fn linspace(n: usize) -> Vec<f64> {
    assert!(n >= 2, "n must be at least 2");
    let step = 1.0 / (n - 1) as f64;
    (0..n).map(|i| i as f64 * step).collect()
}
