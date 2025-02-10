use enterpolation::{linear::Linear, Generator};
use ndarray::Array3;
use ndarray_images::Image;
use palette::{LinSrgb, Srgb};
use serde::{Deserialize, Serialize};

use mandybrot::{render_fractal, Complex, Fractal};

type Precision = f32;

const INPUT_DIR: &str = "input";
const OUTPUT_DIR: &str = "output";

#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters<T> {
    pub centre: Complex<T>,
    pub scale: T,
    pub resolution: [u32; 2],

    pub max_iter: u32,
    pub super_samples: u32,
    pub fractal: Fractal<T>,
    pub gamma: T,

    pub colours: Vec<String>,
    pub image_name: String,
}

fn main() {
    // Read parameters from file
    let params = read_input_args();

    // Render the attractor
    let data = render_fractal(
        params.centre,
        params.max_iter,
        params.scale,
        params.resolution,
        params.fractal,
        params.super_samples,
    );

    // Convert iteration counts to normalised values (0.0 - 1.0)
    let min = *data.iter().min().unwrap() as Precision;
    let max = *data.iter().max().unwrap() as Precision;
    let range = (max - min) as Precision;
    let data = data.mapv(|v| (v as Precision - min) / range);

    // Apply gamma correction
    let data = data.mapv(|v| v.powf(params.gamma as Precision));

    // Create a colour map
    let cmap = build_colour_map(&params.colours);

    // Apply the colour map to convert greyscale values to RGB
    let coloured_data = data.mapv(|v| cmap.gen(v));

    // Convert from `Array2<LinSrgb<Precision>>` to `Array3<Precision>`
    let (height, width) = coloured_data.dim();
    let data: Array3<Precision> = Array3::from_shape_fn((height, width, 3), |(y, x, channel)| {
        let pixel = &coloured_data[(y, x)];
        match channel {
            0 => pixel.red,
            1 => pixel.green,
            2 => pixel.blue,
            _ => unreachable!(),
        }
    });

    // Save the image
    let filename = format!("{}/{}", OUTPUT_DIR, params.image_name);
    data.save(filename).unwrap();
}

fn read_input_args() -> Parameters<Precision> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <parameters file>", args[0]);
        std::process::exit(1);
    }
    let params_file = &args[1];

    serde_yaml::from_str(
        &std::fs::read_to_string(format!("{}/{}", INPUT_DIR, params_file)).unwrap(),
    )
    .unwrap()
}

fn hex_to_lin_srgb(hex: &str) -> LinSrgb<Precision> {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).expect("Invalid hex code");
    let g = u8::from_str_radix(&hex[2..4], 16).expect("Invalid hex code");
    let b = u8::from_str_radix(&hex[4..6], 16).expect("Invalid hex code");
    Srgb::new(
        r as Precision / 255.0,
        g as Precision / 255.0,
        b as Precision / 255.0,
    )
    .into_linear()
}

fn linspace(n: usize) -> Vec<Precision> {
    assert!(n >= 2, "n must be at least 2");
    let step = 1.0 / (n - 1) as Precision;
    (0..n).map(|i| i as Precision * step).collect()
}

fn build_colour_map(
    colour_hexes: &[String],
) -> impl Generator<Precision, Output = LinSrgb<Precision>> {
    let colours: Vec<LinSrgb<Precision>> = colour_hexes
        .iter()
        .map(|hex| hex_to_lin_srgb(hex))
        .collect();
    let num_colours = colours.len();
    Linear::builder()
        .elements(colours)
        .knots(linspace(num_colours))
        .build()
        .expect("Failed to build gradient.")
}
