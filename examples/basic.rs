use enterpolation::{linear::ConstEquidistantLinear, Generator};
use palette::LinSrgb;

use mandybrot::{sample_area, Complex};

fn main() {
    let gradient = ConstEquidistantLinear::<f32, _, 3>::equidistant_unchecked([
        LinSrgb::new(0.00, 0.05, 0.20),
        LinSrgb::new(0.70, 0.10, 0.20),
        LinSrgb::new(0.95, 0.90, 0.30),
    ]);

    let x = 0.55;
    let color = gradient.gen(x);
    println!("{} {} {}", color.red, color.green, color.blue);

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [20, 20];
    let data = sample_area(centre, max_iter, scale, resolution);

    let rows = data.shape()[0];
    for y in 0..rows {
        let row = data.row(y);
        for elem in row {
            print!("{:3} ", elem);
        }
        println!();
    }
}
