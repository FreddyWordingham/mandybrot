use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

impl<T> Complex<T> {
    pub const fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
}

/// Negation
impl<T: Neg<Output = T> + Copy> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.real, -self.imag)
    }
}

/// Complex addition
impl<T: Copy + Add<Output = T>> Add for Complex<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

/// Complex subtraction
impl<T: Copy + Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }
}

/// Scalar division
impl Complex<f32> {
    pub fn div_scalar(self, scalar: f32) -> Self {
        Self {
            real: self.real / scalar,
            imag: self.imag / scalar,
        }
    }
}

/// Complex division
impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>> Div
    for Complex<T>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            real: (self.real * other.real + self.imag * other.imag)
                / (other.real * other.real + other.imag * other.imag),
            imag: (self.imag * other.real - self.real * other.imag)
                / (other.real * other.real + other.imag * other.imag),
        }
    }
}

/// Scalar multiplication
impl<T: Copy + Div<Output = T>> Div<T> for Complex<T> {
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self {
            real: self.real / scalar,
            imag: self.imag / scalar,
        }
    }
}

/// Complex multiplication
impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

/// Norm
impl Complex<f32> {
    pub fn norm(&self) -> f32 {
        self.norm_sqr().sqrt()
    }
}

// Norm squared
impl<T: Copy + Add<Output = T> + Mul<Output = T>> Complex<T> {
    pub fn norm_sqr(&self) -> T {
        self.real * self.real + self.imag * self.imag
    }
}

/// Integer power
impl<T: Float> Complex<T> {
    pub fn powi(self, n: u32) -> Self {
        if n == 0 {
            return Self::new(T::one(), T::zero());
        }
        let mut result = self;
        for _ in 1..n {
            result = result * self;
        }
        result
    }
}

/// Float power
impl Complex<f32> {
    pub fn powf(self, n: f32) -> Self {
        let r = self.norm();
        let theta = self.imag.atan2(self.real);
        let new_r = r.powf(n);
        let new_theta = theta * n;
        Self::new(new_r * new_theta.cos(), new_r * new_theta.sin())
    }
}

/// Absolute value
impl<T: Float> Complex<T> {
    pub fn abs(self) -> T {
        self.norm_sqr().sqrt()
    }

    // Reciprocal/inverse
    pub fn inv(self) -> Self {
        let norm = self.norm_sqr();
        Self::new(self.real / norm, -self.imag / norm)
    }
}
