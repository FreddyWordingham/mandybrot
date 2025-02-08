use ndarray_images::Image;

use mandybrot::{sample_area, Complex};

const OUTPUT_DIR: &str = "output";
const FILENAME: &str = "grayscale.png";

fn main() {
    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [2048, 2048];
    let data = sample_area(centre, max_iter, scale, resolution);

    // Convert to normalised f32 values
    let data = data.mapv(|v| v as f32 / max_iter as f32);

    // Create an image from the data
    let filename = format!("{}/{}", OUTPUT_DIR, FILENAME);
    data.save(filename).unwrap();
}
