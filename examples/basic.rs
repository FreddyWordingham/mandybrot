use mandybrot::{render_fractal, Complex, Fractal};

fn main() {
    let fractal = Fractal::Mandelbrot;

    let centre = Complex::new(-0.75, 0.0);
    let max_iter = 100;
    let scale = 3.0;
    let resolution = [21, 21];
    let super_samples = 1;
    let data = render_fractal(centre, max_iter, scale, resolution, fractal, super_samples);

    let rows = data.shape()[0];
    for y in 0..rows {
        let row = data.row(y);
        for elem in row {
            print!("{:3} ", elem);
        }
        println!();
    }
}
