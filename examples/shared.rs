use enterpolation::{linear::Linear, Generator};
use palette::{LinSrgba, Srgba};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::read_to_string};

type Precision = f32;

const INPUT_DIR: &str = "input";
pub const OUTPUT_DIR: &str = "output";

#[derive(Debug, Serialize, Deserialize)]
pub struct ColourMaps(HashMap<String, Vec<String>>);

pub fn read_input_args<Parameters>() -> Parameters
where
    for<'de> Parameters: Deserialize<'de>,
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <parameters file>", args[0]);
        std::process::exit(1);
    }
    let params_file = &args[1];
    let params_filepath = format!("{}/{}", INPUT_DIR, params_file);
    let file_contents = read_to_string(&params_filepath).expect(&format!(
        "Failed to read parameters file: {}",
        params_filepath
    ));
    serde_yaml::from_str(&file_contents).expect(&format!(
        "Failed to parse parameters file: {}",
        params_filepath
    ))
}

pub fn create_colour_map(
    colour_map_name: &str,
) -> impl Generator<Precision, Output = LinSrgba<Precision>> {
    let cmap_filepath = format!("{}/colour_maps.yaml", INPUT_DIR);
    let colour_maps: ColourMaps = serde_yaml::from_str(&read_to_string(&cmap_filepath).expect(
        &format!("Failed to read colour maps file: {}", cmap_filepath),
    ))
    .expect(&format!(
        "Failed to parse colour maps file: {}",
        cmap_filepath
    ));

    let colour_map = colour_maps
        .0
        .get(colour_map_name)
        .expect(&format!("Colour map '{}' not found.", colour_map_name));
    build_colour_map(colour_map)
}

fn hex_to_lin_srgba(hex: &str) -> LinSrgba<Precision> {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).expect(&format!("Invalid hex code: {}", hex));
    let g = u8::from_str_radix(&hex[2..4], 16).expect(&format!("Invalid hex code: {}", hex));
    let b = u8::from_str_radix(&hex[4..6], 16).expect(&format!("Invalid hex code: {}", hex));
    let a = if hex.len() == 8 {
        u8::from_str_radix(&hex[6..8], 16).expect(&format!("Invalid hex code: {}", hex))
    } else {
        255
    };
    Srgba::new(
        r as Precision / 255.0,
        g as Precision / 255.0,
        b as Precision / 255.0,
        a as Precision / 255.0,
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
) -> impl Generator<Precision, Output = LinSrgba<Precision>> {
    let colours: Vec<LinSrgba<Precision>> = colour_hexes
        .iter()
        .map(|hex| hex_to_lin_srgba(hex))
        .collect();
    let num_colours = colours.len();
    Linear::builder()
        .elements(colours)
        .knots(linspace(num_colours))
        .build()
        .expect("Failed to build gradient.")
}
