use num_traits::{Float, NumCast};
use std::ops::{Add, Mul, Sub};

use crate::Complex;

/// Enum representing different fractals that can be sampled.
pub enum Fractal<T> {
    Mandelbrot,
    BurningShip,
    Julia { c: Complex<T> },
    Tricorn,
    Multibrot { power: u32 },
    Newton { epsilon: T },
    Phoenix { c: Complex<T> },
    Clifford { a: T, b: T, c: T, d: T },
    CelticMandelbrot,
}

impl<T> Fractal<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialOrd + Float + NumCast,
{
    /// Samples a given fractal at the provided complex coordinate.
    pub fn sample(&self, p: Complex<T>, max_iter: u32) -> u32 {
        match self {
            Fractal::Mandelbrot => mandelbrot(p, max_iter),
            Fractal::BurningShip => burning_ship(p, max_iter),
            Fractal::Julia { c } => julia(p, *c, max_iter),
            Fractal::Tricorn => tricorn(p, max_iter),
            Fractal::Multibrot { power } => multibrot(p, *power, max_iter),
            Fractal::Newton { epsilon } => newton(p, *epsilon, max_iter),
            Fractal::Phoenix { c } => phoenix(p, *c, max_iter),
            Fractal::Clifford { a, b, c, d } => clifford(p, *a, *b, *c, *d, max_iter),
            Fractal::CelticMandelbrot => celtic_mandelbrot(p, max_iter),
        }
    }
}

#[inline(always)]
fn mandelbrot<T>(c: Complex<T>, max_iter: u32) -> u32
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + PartialOrd + NumCast,
{
    let zero = NumCast::from(0).unwrap();
    let four = NumCast::from(4).unwrap();

    let mut z = Complex::new(zero, zero);
    let mut n = 0;

    while z.norm_sqr() < four && n < max_iter {
        let zz = z * z;
        z = zz + c;
        n += 1;
    }

    n
}
#[inline(always)]
fn burning_ship<T>(c: Complex<T>, max_iter: u32) -> u32
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Float + PartialOrd + NumCast, // Add NumCast for explicit conversions
{
    let mut z = Complex::new(T::from(0.0).unwrap(), T::from(0.0).unwrap());
    let mut iter = 0;

    while z.norm_sqr() < T::from(4.0).unwrap() && iter < max_iter {
        z = Complex::new(z.real.abs(), z.imag.abs());
        z = z * z + c;
        iter += 1;
    }

    iter
}

#[inline(always)]
fn julia<T>(z: Complex<T>, c: Complex<T>, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let four = T::from(4.0).unwrap();
    let mut z = z;
    let mut n = 0;

    while z.norm_sqr() < four && n < max_iter {
        z = z * z + c;
        n += 1;
    }

    n
}

#[inline(always)]
pub fn tricorn<T>(c: Complex<T>, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let zero = T::zero();
    let four = T::from(4.0).unwrap();
    let mut z = Complex::new(zero, zero);
    let mut n = 0;

    while z.norm_sqr() < four && n < max_iter {
        z = Complex::new(z.real, -z.imag) * Complex::new(z.real, -z.imag) + c;
        n += 1;
    }

    n
}

#[inline(always)]
pub fn multibrot<T>(c: Complex<T>, power: u32, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let zero = T::zero();
    let four = T::from(4.0).unwrap();
    let mut z = Complex::new(zero, zero);
    let mut n = 0;

    while z.norm_sqr() < four && n < max_iter {
        z = z.powi(power) + c;
        n += 1;
    }

    n
}

#[inline(always)]
pub fn newton<T>(c: Complex<T>, epsilon: T, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let mut z = c;
    let mut n = 0;

    while n < max_iter {
        let f = z * z * z - Complex::new(T::one(), T::zero());
        let df = Complex::new(T::from(3.0).unwrap(), T::zero()) * z * z;
        let dz = f / df;
        z = z - dz;

        if dz.norm_sqr() < epsilon {
            break;
        }

        n += 1;
    }

    n
}

#[inline(always)]
pub fn phoenix<T>(p: Complex<T>, c: Complex<T>, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let mut z = Complex::new(T::zero(), T::zero());
    let mut z_old = z;
    let mut n = 0;

    while z.norm_sqr() < T::from(4.0).unwrap() && n < max_iter {
        let temp = z;
        z = z * z + c * z_old + p;
        z_old = temp;
        n += 1;
    }

    n
}

#[inline(always)]
pub fn clifford<T>(p: Complex<T>, a: T, b: T, c: T, d: T, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let mut z = p;
    let mut n = 0;

    while z.norm_sqr() < T::from(4.0).unwrap() && n < max_iter {
        z = Complex::new(
            (a * z.imag).sin() + c * (a * z.real).cos(),
            (b * z.real).sin() + d * (b * z.imag).cos(),
        );
        n += 1;
    }

    n
}

#[inline(always)]
fn celtic_mandelbrot<T>(c: Complex<T>, max_iter: u32) -> u32
where
    T: Float + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    let zero = T::zero();
    let four = T::from(4.0).unwrap();
    let mut z = Complex::new(zero, zero);
    let mut n = 0;
    while z.norm_sqr() < four && n < max_iter {
        // Absolute value applied to the real part difference
        z = Complex::new(
            (z.real * z.real - z.imag * z.imag).abs(),
            T::from(2.0).unwrap() * z.real * z.imag,
        ) + c;
        n += 1;
    }
    n
}
