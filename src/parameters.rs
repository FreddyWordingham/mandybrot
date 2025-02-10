use serde::{Deserialize, Serialize};

use crate::{Attractor, Complex};

#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters<T> {
    pub start: Complex<T>,
    pub centre: Complex<T>,
    pub max_iter: u32,
    pub scale: T,
    pub resolution: [u32; 2],
    pub colours: Vec<String>,
    pub attractor: Attractor<T>,
    pub gamma: T,
}
