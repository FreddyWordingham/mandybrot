use ndarray::Array2;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

mod complex;
mod mandelbrot;

pub use complex::Complex;
pub use mandelbrot::mandelbrot;

pub fn sample_area<T>(
    centre: Complex<T>,
    max_iter: u32,
    scale: T,
    resolution: [u32; 2],
) -> Array2<u32>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + From<u32>
        + Send
        + Sync,
{
    let [x_res, y_res] = resolution;

    // Compute aspect ratio and scale width/height accordingly
    let aspect_ratio = T::from(x_res) / T::from(y_res);
    let width = scale * aspect_ratio;
    let height = scale;

    let x_res_t = T::from(x_res);
    let y_res_t = T::from(y_res);
    let half_x_res = x_res_t / T::from(2);
    let half_y_res = y_res_t / T::from(2);

    let x_step = width / x_res_t;
    let y_step = height / y_res_t;

    let mut samples = Array2::<u32>::zeros((y_res as usize, x_res as usize));

    // Parallel row processing with direct mutable chunks
    samples
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(x_res as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_t = T::from(y as u32);
            let y_offset = (y_t - half_y_res) * y_step;
            let y_coord = centre.imag + y_offset;

            row.iter_mut().enumerate().for_each(|(x, elem)| {
                let x_t = T::from(x as u32);
                let x_coord = centre.real + (x_t - half_x_res) * x_step;
                let c = Complex::new(x_coord, y_coord);

                *elem = mandelbrot(c, max_iter);
            });
        });

    samples
}
