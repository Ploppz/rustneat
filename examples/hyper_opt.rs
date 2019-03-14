extern crate rand;
extern crate rustneat;
#[macro_use]
extern crate blackbox;

use rustneat::{Environment, Organism, Population, NeuralNetwork, Params};
use chrono::{Timelike, Utc};

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut NeuralNetwork) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).powi(2);
        organism.activate(vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 1f64], &mut output);
        distance += (0f64 - output[0]).powi(2);

        let fitness = 16.0 / (1.0 + distance);

        fitness
    }
}

fn run(p: &Params, n_gen: usize) -> f64 {

    let mut start_genome = NeuralNetwork::with_neurons(3);
    start_genome.add_connection(0, 2, 0.0);
    start_genome.add_connection(1, 2, 0.0);
    let mut population = Population::create_population_from(start_genome, 150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    for i in 0..n_gen {
        population.evolve(&mut environment, p);
    }

    let mut best_fitness = 0.0;
    for organism in population.get_organisms() {
        if organism.fitness > best_fitness {
            best_fitness = organism.fitness;
        }
    }
    best_fitness
}

make_optimizer! {
    Configuration {
        prune_after_n_generations: usize = 10 .. 40,
        n_to_prune: usize = 2 .. 4,

        mutation_pr: f64 = 0.2 .. 1.0,
        interspecie_mate_pr: f64 = 0.0 .. 0.002,
        cull_fraction: f64 = 0.05 .. 0.3,

        c2: f64 = 0.3 .. 1.0,
        c3: f64 = 0.0 .. 0.4,
        mutate_conn_weight_pr: f64 = 0.2 .. 0.9,
        mutate_conn_weight_perturbed_pr: f64 = 0.2 .. 0.9,
        // n_conn_to_mutate: 0,
        mutate_add_conn_pr: f64 = 0.001..0.004,
        // mutate_add_neuron_pr: f64 = 0.001..0.002,
        mutate_toggle_expr_pr: f64 = 0.001 .. 0.02,
        mutate_bias_pr: f64 = 0.01 .. 0.05,
        include_weak_disjoint_gene: f64 = 0.1 .. 0.3,

        compatibility_threshold: f64 = 2.0 .. 4.0,
    }

    const N_GEN: usize = 30; // generations per round
    const N_POPULATIONS: usize = 30; // populations per iteration
    let p = Params {
        prune_after_n_generations,
        n_to_prune,

        mutation_pr,
        interspecie_mate_pr,
        cull_fraction,

        c2,
        c3,
        mutate_conn_weight_pr,
        mutate_conn_weight_perturbed_pr,
        n_conn_to_mutate: 0,
        mutate_add_conn_pr,
        mutate_add_neuron_pr: 0.001,
        mutate_toggle_expr_pr,
        mutate_bias_pr,
        include_weak_disjoint_gene,

        compatibility_threshold,
    };
    // Take the average of N rounds
    let score = (0..N_POPULATIONS)
        .map(|_| run(&p, N_GEN))
        .sum::<f64>() / N_POPULATIONS as f64;
    println!("Iteration... Score = {}", score);
    score
    
}
fn main() {
    let now = Utc::now();
    println!("Start: {:02}:{:02}:{:02}", now.hour(), now.minute(), now.second());

    const N_ITER: usize = 300;
    let config = Configuration::random_search(N_ITER);
    println!("Score: {}", config.evaluate());
    println!("Config: {:?}", config);

    println!("\nExecution time: {}", (Utc::now() - now).num_seconds());
}