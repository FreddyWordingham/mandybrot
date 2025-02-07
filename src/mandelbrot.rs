use std::ops::{Add, Mul, Sub};

use crate::Complex;

#[inline(always)]
pub fn mandelbrot<T>(c: Complex<T>, max_iter: u32) -> u32
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + PartialOrd + From<u32>,
{
    let zero = T::from(0);
    let four = T::from(4);

    let mut z = Complex::new(zero, zero);
    let mut n = 0;

    while z.norm_sqr() < four && n < max_iter {
        let zz = z * z;
        z = zz + c;
        n += 1;
    }

    n
}
