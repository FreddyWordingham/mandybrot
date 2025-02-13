use enterpolation::Generator;
use ndarray::{Array2, Array3, Zip};
use ndarray_images::Image;
use palette::Darken;
use serde::{Deserialize, Serialize};

use mandybrot::{render_fractal, Complex, Fractal};

mod shared;
use shared::{create_colour_map, read_input_args, OUTPUT_DIR};

type Precision = f64;

#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters<T> {
    pub centre: [T; 2],
    pub scale: T,
    pub resolution: [u32; 2],
    pub super_samples: u32,

    pub max_iter: u32,
    pub light_dir: [T; 3],

    pub fractal: Fractal<T>,

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
    let data = render_fractal(
        Complex::new(params.centre[0], params.centre[1]),
        params.max_iter,
        params.scale,
        params.resolution,
        params.fractal,
        params.super_samples,
    );
    let shadow_map = create_shadow_map(&data, &params.light_dir);
    // let ao_map = create_ambient_occlusion_map(
    //     &data, 16, 16, 1.0e-1, // params.scale / params.resolution[0] as Precision,
    // );
    let ao_map = create_ambient_occlusion_map(
        &data, 4, 4, 1.0e-1, // params.scale / params.resolution[0] as Precision,
    );
    let shadow_map = shadow_map * &ao_map;
    // let shadow_map = ao_map;

    // Normalise the data
    let max = *data.iter().max().unwrap() as Precision;
    let data = if params.log {
        data.mapv(|v| (v as Precision).ln().max(0.0) / (max as Precision).ln())
    } else {
        data.mapv(|v| v as Precision / max as Precision)
    };

    // Apply gamma correction
    let data = data.mapv(|v| v.powf(params.gamma as Precision));

    // Create colours from samples plus shadow map
    let coloured_data = Zip::from(&data).and(&shadow_map).map_collect(|&v, &s| {
        let colour = cmap.gen(v as f32);
        colour.darken(s as f32)
    });

    // Convert from `Array2<LinSrgb<Precision>>` to `Array3<Precision>`
    let (height, width) = coloured_data.dim();
    let data: Array3<f32> = Array3::from_shape_fn((height, width, 3), |(y, x, channel)| {
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

fn create_shadow_map(samples: &Array2<u32>, light_dir: &[Precision; 3]) -> Array2<Precision> {
    let (height, width) = samples.dim();
    let mut shadow_map = Array2::<Precision>::zeros((height, width));

    // Normalize the light direction
    let norm =
        (light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2])
            .sqrt();
    let light = (
        light_dir[0] / norm,
        light_dir[1] / norm,
        light_dir[2] / norm,
    );

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Compute central differences as gradients.
            let dzdx = (samples[(y, x + 1)] as Precision - samples[(y, x - 1)] as Precision) * 0.5;
            let dzdy = (samples[(y + 1, x)] as Precision - samples[(y - 1, x)] as Precision) * 0.5;

            // Approximate surface normal: N = (-dzdx, -dzdy, 1)
            let nx = -dzdx;
            let ny = -dzdy;
            let nz = 1.0;
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            let n = (nx / norm, ny / norm, nz / norm);

            // Dot product with light direction
            let intensity = n.0 * light.0 + n.1 * light.1 + n.2 * light.2;
            shadow_map[(y, x)] = intensity.max(0.0);
        }
    }
    shadow_map
}

fn create_ambient_occlusion_map(
    samples: &Array2<u32>,
    num_angles: usize,
    max_radius: usize,
    pixel_size: Precision, // real-world size per pixel
) -> Array2<Precision> {
    let (height, width) = samples.dim();
    let mut ao_map = Array2::<Precision>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let h0 = samples[(y, x)] as Precision;
            let mut total = 0.0;
            for i in 0..num_angles {
                let theta = 2.0
                    * Precision::from(std::f32::consts::PI)
                    * (i as Precision / num_angles as Precision);
                let mut max_angle = -Precision::from(std::f32::consts::PI) / 2.0;
                for r in 1..=max_radius {
                    let nx = x as isize + (r as Precision * theta.cos()).round() as isize;
                    let ny = y as isize + (r as Precision * theta.sin()).round() as isize;
                    if nx < 0 || nx >= width as isize || ny < 0 || ny >= height as isize {
                        break;
                    }
                    let h = samples[(ny as usize, nx as usize)] as Precision;
                    let distance = r as Precision * pixel_size;
                    let sample_angle = ((h - h0) / distance).atan();
                    if sample_angle > max_angle {
                        max_angle = sample_angle;
                    }
                }
                // Unoccluded contribution: if max_angle is negative, light is fully visible.
                let contribution = if max_angle < 0.0 {
                    1.0
                } else {
                    max_angle.cos()
                };
                total += contribution;
            }
            ao_map[(y, x)] = 1.0 - (total / num_angles as Precision);
        }
    }
    ao_map
}
