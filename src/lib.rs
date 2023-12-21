use pyo3::prelude::*;
use pyo3::exceptions::*;
use pyo3::types::PyList;

const RANDMAX: u64 = u64::MAX;

struct XorShift128Plus {
    state: [u64; 2],
}

impl XorShift128Plus {
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, RANDMAX);
        XorShift128Plus { state: [seed, 0] }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state[0];
        let y = self.state[1];
        self.state[0] = y;
        x ^= x << 23;
        self.state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        self.state[1].wrapping_add(y)
    }
}

fn get_flip_energy(
    var: usize,
    state: &[i8],
    h: &[f64],
    degrees: &[usize],
    neighbors: &[Vec<usize>],
    neighbor_couplings: &[Vec<f64>],
) -> f64 {
    let mut energy = h[var];
    for n_i in 0..degrees[var] {
        energy += state[neighbors[var][n_i]] as f64 * neighbor_couplings[var][n_i];
    }
    -2.0 * state[var] as f64 * energy
}

#[derive(PartialEq)]
pub enum Proposal {
    Metropolis,
    Gibbs,
}

#[derive(PartialEq)]
pub enum VariableOrder {
    Random,
    Sequential,
}

fn simulated_annealing_run(
    state: &mut [i8],
    h: &[f64],
    degrees: &[usize],
    neighbors: &[Vec<usize>],
    neighbor_couplings: &[Vec<f64>],
    sweeps_per_beta: usize,
    beta_schedule: &[f64],
    seed: u64,
    var_order: VariableOrder,
    proposal_acceptance_criteria: Proposal,
) {
    let num_vars = h.len();

    let mut delta_energy = vec![0.0; num_vars];

    let mut rng = XorShift128Plus::new(seed);

    for var in 0..num_vars {
        delta_energy[var] = get_flip_energy(
            var, state, h, degrees, neighbors, neighbor_couplings
        );
    }

    for &beta in beta_schedule.iter() {
        let threshold = 44.36142 / beta;
        for _ in 0..sweeps_per_beta {
            for var in 0..num_vars {
                let var = match var_order {
                    VariableOrder::Random => rng.next() as usize % num_vars,
                    VariableOrder::Sequential => var,
                };

                let delta_e = delta_energy[var];

                if delta_e >= threshold {
                    println!("Warning: delta_e = {} >= threshold = {}", delta_e, threshold);
                    continue;
                }

                let flip_spin = match proposal_acceptance_criteria {
                    Proposal::Metropolis => {
                        if delta_e <= 0.0 {
                            true
                        } else {
                            let rand = rng.next() as f64 / RANDMAX as f64;
                            (-delta_e * beta).exp() > rand
                        }
                    },
                    Proposal::Gibbs => {
                        let rand = rng.next() as f64 / RANDMAX as f64;
                        1.0 / (1.0 + (-delta_e * beta).exp()) > rand
                    },
                };

                if flip_spin {
                    let multiplier = 4.0 * state[var] as f64;
                    for n_i in 0..degrees[var] {
                        let neighbor = neighbors[var][n_i];
                        delta_energy[neighbor] += multiplier * neighbor_couplings[var][n_i] * state[neighbor] as f64;
                    }
                    state[var] *= -1;
                    delta_energy[var] *= -1.0;
                }
            }
        }
    }
}

fn get_state_energy(
    state: &[i8],
    h: &[f64],
    coupler_starts: &[usize],
    coupler_ends: &[usize],
    coupler_weights: &[f64],
) -> f64 {
    let mut energy = 0.0;

    for (var, &h_val) in h.iter().enumerate() {
        energy += state[var] as f64 * h_val;
    }

    for (&start, (&end, &weight)) in coupler_starts.iter().zip(coupler_ends.iter().zip(coupler_weights.iter())) {
        energy += state[start] as f64 * weight * state[end] as f64;
    }

    energy
}

/// Perform simulated annealing on a general problem.
#[pyfunction]
fn general_simulated_annealing(
    _py: Python,
    states: &PyList,
    energies: &PyList,
    num_samples: usize,
    h: Vec<f64>,
    coupler_starts: Vec<usize>,
    coupler_ends: Vec<usize>,
    coupler_weights: Vec<f64>,
    sweeps_per_beta: usize,
    beta_schedule: Vec<f64>,
    seed: u64,
    varorder: &str,
    proposal_acceptance_criteria: &str,
) -> PyResult<(Vec<i8>, Vec<f64>)> {
    let num_vars = h.len();
    if coupler_starts.len() != coupler_ends.len() || coupler_starts.len() != coupler_weights.len() {
        Err(PyErr::new::<PyException, _>("coupler vectors have mismatched lengths"))?;
    }

    let mut states: Vec<i8> = states.extract()?;
    let mut energies: Vec<f64> = energies.extract()?;

    let _varorder = match varorder.as_ref() {
        "random" => VariableOrder::Random,
        "sequential" => VariableOrder::Sequential,
        _ => Err(PyErr::new::<PyException, _>("invalid variable order"))?,
    };

    let _proposal_acceptance_criteria = match proposal_acceptance_criteria.as_ref() {
        "metropolis" => Proposal::Metropolis,
        "gibbs" => Proposal::Gibbs,
        _ => Err(PyErr::new::<PyException, _>("invalid proposal acceptance criteria"))?,
    };

    let mut degrees = vec![0; num_vars];
    let mut neighbors = vec![Vec::new(); num_vars];
    let mut neighbour_couplings = vec![Vec::new(); num_vars];

    for (&start, (&end, &weight)) in coupler_starts.iter().zip(coupler_ends.iter().zip(coupler_weights.iter())) {
        if start >= num_vars || end >= num_vars {
            Err(PyErr::new::<PyException, _>("coupler indexes contain an invalid variable"))?;
        }

        neighbors[start].push(end);
        neighbors[end].push(start);
        neighbour_couplings[start].push(weight);
        neighbour_couplings[end].push(weight);

        degrees[start] += 1;
        degrees[end] += 1;
    }

    let mut sample = 0;
    while sample < num_samples {
        let state = &mut states[sample * num_vars..(sample + 1) * num_vars];

        if _varorder == VariableOrder::Random {
            if _proposal_acceptance_criteria == Proposal::Metropolis {
                simulated_annealing_run(
                    state, &h, &degrees, &neighbors, &neighbour_couplings, sweeps_per_beta, &beta_schedule, seed + sample as u64, VariableOrder::Random, Proposal::Metropolis
                );
            } else {
                simulated_annealing_run(
                    state, &h, &degrees, &neighbors, &neighbour_couplings, sweeps_per_beta, &beta_schedule, seed + sample as u64, VariableOrder::Random, Proposal::Gibbs
                );
            }
        } else {
            if _proposal_acceptance_criteria == Proposal::Metropolis {
                simulated_annealing_run(
                    state, &h, &degrees, &neighbors, &neighbour_couplings, sweeps_per_beta, &beta_schedule, seed + sample as u64, VariableOrder::Sequential, Proposal::Metropolis
                );
            } else {
                simulated_annealing_run(
                    state, &h, &degrees, &neighbors, &neighbour_couplings, sweeps_per_beta, &beta_schedule, seed + sample as u64, VariableOrder::Sequential, Proposal::Gibbs
                );
            }
        }

        energies[sample] = get_state_energy(state, &h, &coupler_starts, &coupler_ends, &coupler_weights);
        sample += 1;
    }

    Ok((states, energies))
}

/// A Python module implemented in Rust.
#[pymodule]
fn neal_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(general_simulated_annealing, m)?)?;
    Ok(())
}
