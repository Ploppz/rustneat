extern crate rustneat;
extern crate rand;
use rand::distributions::{Distribution, Uniform};
use rustneat::nn::Ctrnn;
use chrono::{Datelike, Timelike, Utc};

fn main() {

    let mut rng = rand::thread_rng();
    const N_NEURONS: usize = 20;
    const N_ITER: usize = 100;
    let uniform = Uniform::from(0.0..1.0);

    let now = Utc::now();

    for i in 0..N_ITER {
        Ctrnn {
            y: &uniform.sample_iter(&mut rng).take(N_NEURONS).collect::<Vec<_>>(),
            delta_t: 1.0,
            tau: &uniform.sample_iter(&mut rng).take(N_NEURONS).collect::<Vec<_>>(),
            wij:&uniform.sample_iter(&mut rng).take(N_NEURONS*N_NEURONS).collect::<Vec<_>>(),
            theta: &uniform.sample_iter(&mut rng).take(N_NEURONS).collect::<Vec<_>>(),
            i: &uniform.sample_iter(&mut rng).take(N_NEURONS).collect::<Vec<_>>(),

        }.activate_nn(10);
    }

    println!("\nExecution time: {}", (Utc::now() - now).num_milliseconds());
}
