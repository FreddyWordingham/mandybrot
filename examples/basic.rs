use mandybrot::{sample_area, Complex, Fractal};

fn main() {
    let fractal = Fractal::Mandelbrot;

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [20, 20];
    let data = sample_area(centre, max_iter, scale, resolution, fractal);

    let rows = data.shape()[0];
    for y in 0..rows {
        let row = data.row(y);
        for elem in row {
            print!("{:3} ", elem);
        }
        println!();
    }
}
