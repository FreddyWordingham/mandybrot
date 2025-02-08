use ndarray::Array2;
use num_traits::{Float, NumCast};
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

use crate::{Complex, Fractal};

/// Generic function to sample a fractal based on the selected FractalType.
pub fn sample_area<T>(
    centre: Complex<T>,
    max_iter: u32,
    scale: T,
    resolution: [u32; 2],
    fractal: Fractal<T>,
) -> Array2<u32>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + NumCast // Ensure NumCast is included in trait bounds
        + Float
        + Send
        + Sync,
{
    let [x_res, y_res] = resolution;

    // Use NumCast to avoid ambiguity
    let x_res_t = T::from(x_res).unwrap();
    let y_res_t = T::from(y_res).unwrap();

    let aspect_ratio = x_res_t / y_res_t;
    let width = scale * aspect_ratio;
    let height = scale;

    let half_x_res = x_res_t / T::from(2).unwrap();
    let half_y_res = y_res_t / T::from(2).unwrap();

    let x_step = width / x_res_t;
    let y_step = height / y_res_t;

    let mut samples = Array2::<u32>::zeros((y_res as usize, x_res as usize));

    samples
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(x_res as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_t = T::from(y as u32).unwrap();
            let y_offset = (y_t - half_y_res) * y_step;
            let y_coord = centre.imag + y_offset;

            row.iter_mut().enumerate().for_each(|(x, elem)| {
                let x_t = T::from(x as u32).unwrap();
                let x_coord = centre.real + (x_t - half_x_res) * x_step;
                let c = Complex::new(x_coord, y_coord);

                *elem = fractal.sample(c, max_iter);
            });
        });

    samples
}

/// Sample a fractal with anti-aliasing.
pub fn multisample_area<T>(
    centre: Complex<T>,
    max_iter: u32,
    scale: T,
    resolution: [u32; 2],
    fractal: Fractal<T>,
    samples: u32,
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

    // Each pixel spans these steps.
    let x_step = width / x_res_t;
    let y_step = height / y_res_t;

    // Used for centering the image.
    let half_x_res = x_res_t / T::from(2).unwrap();
    let half_y_res = y_res_t / T::from(2).unwrap();

    let mut result = Array2::<u32>::zeros((y_res as usize, x_res as usize));

    // Parallelize over rows.
    result
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(x_res as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_t = T::from(y).unwrap();
            // Compute the pixel center's y coordinate.
            let pixel_center_y = centre.imag + (y_t + T::from(0.5).unwrap() - half_y_res) * y_step;
            for (x, pixel) in row.iter_mut().enumerate() {
                let x_t = T::from(x).unwrap();
                // Compute the pixel center's x coordinate.
                let pixel_center_x =
                    centre.real + (x_t + T::from(0.5).unwrap() - half_x_res) * x_step;

                let mut sum = 0u32;
                let samples_t = T::from(samples).unwrap();
                // Loop over a samples x samples grid within the pixel.
                for i in 0..samples {
                    let i_t = T::from(i).unwrap();
                    // Offset in the x direction.
                    let offset_x = ((i_t + T::from(0.5).unwrap()) / samples_t
                        - T::from(0.5).unwrap())
                        * x_step;
                    for j in 0..samples {
                        let j_t = T::from(j).unwrap();
                        // Offset in the y direction.
                        let offset_y = ((j_t + T::from(0.5).unwrap()) / samples_t
                            - T::from(0.5).unwrap())
                            * y_step;
                        let sample_x = pixel_center_x + offset_x;
                        let sample_y = pixel_center_y + offset_y;
                        let c = Complex::new(sample_x, sample_y);
                        sum += fractal.sample(c, max_iter);
                    }
                }
                // Average the samples for anti-aliasing.
                let total_samples = samples * samples;
                *pixel = sum / total_samples;
            }
        });

    result
}
