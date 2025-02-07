use std::ops::{Add, Mul, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex<T> {
    pub real: T,
    pub imag: T,
}

impl<T> Complex<T> {
    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
}

// Ensure T supports necessary arithmetic operations
impl<T: Copy + Add<Output = T> + Mul<Output = T>> Complex<T> {
    pub fn norm_sqr(&self) -> T {
        self.real * self.real + self.imag * self.imag
    }
}

impl<T: Copy + Add<Output = T>> Add for Complex<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}
