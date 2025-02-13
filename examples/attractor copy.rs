use enterpolation::Generator;
use ndarray::Array3;
use ndarray_images::Image;
use palette::LinSrgba;
use serde::{Deserialize, Serialize};
use std::{fs::create_dir_all, path::Path};

use mandybrot::{render_attractor, Attractor, Complex};

mod shared;
use shared::{create_colour_map, read_input_args, OUTPUT_DIR};

type Precision = f32;

#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters<T> {
    pub centre: [T; 2],
    pub scale: T,
    pub resolution: [u32; 2],
    pub super_samples: Option<u32>,

    pub start: [T; 2],
    pub radius: T,
    pub num_samples: u32,
    pub max_iter: u32,
    pub draw_after: u32,

    pub attractor: Attractor<T>,

    pub image_name: String,
    pub log: bool,
    pub gamma: T,
    pub colour_map: String,
}

fn main() {
    // Read parameters from file
    let params = read_input_args::<Parameters<Precision>>();

    // Create the colour map
    let cmap = create_colour_map(&params.colour_map);

    // Render the attractor
    let data = render_attractor(
        Complex::new(params.centre[0], params.centre[1]),
        params.scale,
        [
            params.resolution[0] * params.super_samples.unwrap_or(1),
            params.resolution[1] * params.super_samples.unwrap_or(1),
        ],
        Complex::new(params.start[0], params.start[1]),
        params.radius,
        params.num_samples,
        params.max_iter,
        params.draw_after,
        &params.attractor,
    );

    // Normalise the data
    let max = *data.iter().max().unwrap() as Precision;
    let data = if params.log {
        data.mapv(|v| (v as Precision).ln().max(0.0) / (max as Precision).ln())
    } else {
        data.mapv(|v| v as Precision / max as Precision)
    };

    // Apply gamma correction
    let data = data.mapv(|v| v.powf(params.gamma));

    // Apply the colour map to convert greyscale values to RGB
    let mut coloured_data = data.mapv(|v| cmap.gen(v));

    // Average the super samples
    if let Some(super_samples) = params.super_samples {
        coloured_data = downsample(&coloured_data, super_samples as usize);
    }

    // Convert from `Array2<LinSrgb<Precision>>` to `Array3<Precision>`
    let (height, width) = coloured_data.dim();
    let data: Array3<Precision> = Array3::from_shape_fn((height, width, 4), |(y, x, channel)| {
        let pixel = &coloured_data[(y, x)];
        match channel {
            0 => pixel.red,
            1 => pixel.green,
            2 => pixel.blue,
            3 => pixel.alpha,
            _ => unreachable!(),
        }
    });

    // Save the image
    let filename = format!("{}/{}", OUTPUT_DIR, params.image_name);
    let path = Path::new(&filename);
    create_dir_all(path.parent().unwrap()).unwrap();
    data.save(&filename).unwrap();
    println!("Image saved to '{}'.", filename);
}

use ndarray::Array2;
// use num_traits::Zero;
// use std::ops::Div;

// fn downsample<T: Clone + Div + Zero>(input: &Array2<T>, super_samples: usize) -> Array2<T> {
fn downsample(input: &Array2<LinSrgba>, super_samples: usize) -> Array2<LinSrgba> {
    let (height, width) = input.dim();

    assert!(super_samples > 1, "There must be at least 2 super samples");
    assert!(
        height % super_samples == 0 && width % super_samples == 0,
        "Invalid super_samples"
    );

    let averages: Vec<LinSrgba> = input
        .exact_chunks((super_samples, super_samples))
        .into_iter()
        .map(|chunk| {
            let sum = chunk
                .iter()
                .fold(LinSrgba::new(0.0, 0.0, 0.0, 0.0), |acc, &v| acc + v);
            sum / (super_samples * super_samples) as Precision
        })
        .collect();

    Array2::from_shape_vec((height / super_samples, width / super_samples), averages).unwrap()
}
