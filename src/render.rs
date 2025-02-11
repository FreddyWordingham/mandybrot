use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::Array2;
use num_traits::{Float, FloatConst, NumCast};
use rand::{distr::uniform::SampleUniform, rng, Rng};
use rayon::prelude::*;
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

use crate::{Attractor, Complex, Fractal};

/// Renders a fractal with anti-aliasing by sampling multiple points per pixel.
pub fn render_fractal<T>(
    centre: Complex<T>,
    max_iter: u32,
    scale: T,
    resolution: [u32; 2],
    fractal: Fractal<T>,
    samples_per_pixel: u32,
) -> Array2<u32>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NumCast
        + Float
        + Send
        + Sync,
{
    let [x_res, y_res] = resolution;
    let x_res_t = T::from(x_res).unwrap();
    let y_res_t = T::from(y_res).unwrap();
    let aspect_ratio = x_res_t / y_res_t;
    let width = scale * aspect_ratio;
    let height = scale;
    let x_step = width / x_res_t;
    let y_step = height / y_res_t;
    let half_x_res = x_res_t / T::from(2).unwrap();
    let half_y_res = y_res_t / T::from(2).unwrap();

    let mut pixels = Array2::<u32>::zeros((y_res as usize, x_res as usize));

    // Create a progress bar for rendering rows.
    let pb = ProgressBar::new(y_res as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} ETA: {eta}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );

    pixels
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(x_res as usize)
        .enumerate()
        .progress_with(pb)
        .for_each(|(y, row)| {
            let y_t = T::from(y).unwrap();
            let pixel_center_y = centre.imag + (y_t + T::from(0.5).unwrap() - half_y_res) * y_step;
            for (x, pixel) in row.iter_mut().enumerate() {
                let x_t = T::from(x).unwrap();
                let pixel_center_x =
                    centre.real + (x_t + T::from(0.5).unwrap() - half_x_res) * x_step;
                let mut sum = 0u32;
                let samples_t = T::from(samples_per_pixel).unwrap();
                for i in 0..samples_per_pixel {
                    let i_t = T::from(i).unwrap();
                    let offset_x = ((i_t + T::from(0.5).unwrap()) / samples_t
                        - T::from(0.5).unwrap())
                        * x_step;
                    for j in 0..samples_per_pixel {
                        let j_t = T::from(j).unwrap();
                        let offset_y = ((j_t + T::from(0.5).unwrap()) / samples_t
                            - T::from(0.5).unwrap())
                            * y_step;
                        let sample_x = pixel_center_x + offset_x;
                        let sample_y = pixel_center_y + offset_y;
                        let c = Complex::new(sample_x, sample_y);
                        sum += fractal.sample(c, max_iter);
                    }
                }
                let total_samples = samples_per_pixel * samples_per_pixel;
                *pixel = sum / total_samples;
            }
        });

    pixels
}

fn create_position_to_pixel_mapper<T: Float + NumCast + Display>(
    offset: Complex<T>,
    scale: T,
    resolution: [u32; 2],
) -> impl Fn(&Complex<T>) -> Option<[usize; 2]> {
    let x_res = T::from(resolution[0]).unwrap();
    let y_res = T::from(resolution[1]).unwrap();
    let aspect_ratio = x_res / y_res;
    let width = scale * aspect_ratio;
    let height = scale;
    let half_width = width / T::from(2.0).unwrap();
    let half_height = height / T::from(2.0).unwrap();
    let max_x = x_res - T::one();
    let max_y = y_res - T::one();

    move |p: &Complex<T>| {
        // Shift the point by the offset to recenter the image.
        let shifted_real = p.real - offset.real;
        let shifted_imag = p.imag - offset.imag;
        let x = ((shifted_real + half_width) / width) * max_x;
        let y = ((half_height - shifted_imag) / height) * max_y;

        if x >= T::zero() && x < x_res && y >= T::zero() && y < y_res {
            Some([x.to_usize().unwrap(), y.to_usize().unwrap()])
        } else {
            None
        }
    }
}

fn generate_initial_positions<T>(start: Complex<T>, radius: T, num_samples: u32) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast + SampleUniform,
{
    let mut rng = rng();
    let mut positions = Vec::with_capacity(num_samples as usize);
    let zero = T::from(0.0).unwrap();
    let tau = T::TAU();
    for _ in 0..num_samples {
        let theta = rng.random_range(zero..tau);
        let rho = rng.random_range(zero..radius).sqrt();
        let x = start.real + rho * theta.cos();
        let y = start.imag + rho * theta.sin();
        positions.push(Complex::new(x, y));
    }
    positions
}

pub fn render_attractor<T>(
    centre: Complex<T>,
    scale: T,
    resolution: [u32; 2],

    start: Complex<T>,
    radius: T,
    num_samples: u32,

    max_iter: u32,
    draw_after: u32,
    attractor: &Attractor<T>,
) -> Array2<u32>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NumCast
        + Float
        + FloatConst
        + SampleUniform
        + Send
        + Sync
        + Display,
{
    let initial_positions = generate_initial_positions(start, radius, num_samples);

    // Render and sum attractors concurrently.
    let pb = ProgressBar::new(initial_positions.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} ETA: {eta}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );

    let shape = (resolution[1] as usize, resolution[0] as usize);
    initial_positions
        .par_iter()
        .progress_with(pb)
        .map(|&pos| {
            render_attractor_path(
                pos, centre, max_iter, draw_after, scale, resolution, &attractor,
            )
        })
        .reduce(|| Array2::zeros(shape), |a, b| a + b)
}

/// Renders a single part of a point orbiting an attractor by iterating its dynamics and accumulating hits in a pixel grid.
fn render_attractor_path<T>(
    start: Complex<T>,
    centre: Complex<T>,
    max_iter: u32,
    draw_after: u32,
    scale: T,
    resolution: [u32; 2],
    attractor: &Attractor<T>,
) -> Array2<u32>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NumCast
        + Float
        + Send
        + Sync
        + std::fmt::Display,
{
    let [x_res, y_res] = resolution;
    let mut pixels = Array2::<u32>::zeros((y_res as usize, x_res as usize));
    let pixel_mapper = create_position_to_pixel_mapper(centre, scale, resolution);

    let mut pos = start;
    for n in 0..max_iter {
        pos = attractor.iterate(pos);

        if n < draw_after {
            continue;
        }
        if let Some([x, y]) = pixel_mapper(&pos) {
            pixels[[y, x]] += 1;
        }
    }

    pixels
}
