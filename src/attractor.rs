use num_traits::{Float, NumCast, One};
use std::ops::{Add, Mul, Sub};

use crate::Complex;

/// Enum representing different attractors that can be iterated.
pub enum Attractor<T> {
    Clifford { a: T, b: T, c: T, d: T },
    DeJong { a: T, b: T, c: T, d: T },
    Henon { a: T, b: T },
    Ikeda { u: T },
    Tinkerbell { a: T, b: T, c: T, d: T },
}

impl<T> Attractor<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + PartialOrd + Float + NumCast,
{
    /// Iterates the attractor function starting at the provided complex coordinate.
    pub fn iterate(&self, p: Complex<T>) -> Complex<T> {
        match self {
            Attractor::Clifford { a, b, c, d } => clifford(p, *a, *b, *c, *d),
            Attractor::DeJong { a, b, c, d } => de_jong(p, *a, *b, *c, *d),
            Attractor::Henon { a, b } => henon(p, *a, *b),
            Attractor::Ikeda { u } => ikeda(p, *u),
            Attractor::Tinkerbell { a, b, c, d } => tinkerbell(p, *a, *b, *c, *d),
        }
    }
}

#[inline(always)]
fn clifford<T>(p: Complex<T>, a: T, b: T, c: T, d: T) -> Complex<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Float + NumCast,
{
    let x = p.real;
    let y = p.imag;
    Complex {
        real: (a * y).sin() + c * (a * x).cos(),
        imag: (b * x).sin() + d * (b * y).cos(),
    }
}

#[inline(always)]
fn de_jong<T>(p: Complex<T>, a: T, b: T, c: T, d: T) -> Complex<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Float + NumCast,
{
    let x = p.real;
    let y = p.imag;
    Complex {
        real: (a * y).sin() - (b * x).cos(),
        imag: (c * x).sin() - (d * y).cos(),
    }
}

#[inline(always)]
fn henon<T>(p: Complex<T>, a: T, b: T) -> Complex<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Float + NumCast + One,
{
    let x = p.real;
    let y = p.imag;
    Complex {
        real: T::one() - a * x * x + y,
        imag: b * x,
    }
}

#[inline(always)]
fn ikeda<T>(p: Complex<T>, u: T) -> Complex<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float + NumCast + One,
{
    let x = p.real;
    let y = p.imag;
    let r_sq = x * x + y * y;
    let one = T::one();
    let t = T::from(0.4).unwrap() - T::from(6.0).unwrap() / (one + r_sq);
    let cos_t = t.cos();
    let sin_t = t.sin();
    Complex {
        real: one + u * (x * cos_t - y * sin_t),
        imag: u * (x * sin_t + y * cos_t),
    }
}

#[inline(always)]
fn tinkerbell<T>(p: Complex<T>, a: T, b: T, c: T, d: T) -> Complex<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Float + NumCast,
{
    let x = p.real;
    let y = p.imag;
    Complex {
        real: x * x - y * y + a * x + b * y,
        imag: T::from(2.0).unwrap() * x * y + c * x + d * y,
    }
}
