import random

from neal import general_simulated_annealing


def test_general_simulated_annealing():
    # Given

    # When
    num_samples = 5
    num_vars = 4

    states = [
        random.choice([1, -1]) for _ in range(num_vars * num_samples)
    ]
    energies = [0.0] * num_samples

    print(f"initial states: {states}")

    h = [0.5, 0.2, -0.3, 0.1]
    coupler_starts = [0, 1, 2]
    coupler_ends = [1, 2, 3]
    coupler_weights = [1.0, -0.5, 0.5]
    sweeps_per_beta = 100;
    beta_schedule = [1.0, 0.5, 0.1]
    seed = 1234
    varorder = "random"
    proposal_acceptance_criteria = "metropolis"

    result = general_simulated_annealing(
        states,
        energies,
        num_samples,
        h,
        coupler_starts,
        coupler_ends,
        coupler_weights,
        sweeps_per_beta,
        beta_schedule,
        seed,
        varorder,
        proposal_acceptance_criteria,
    )
    states, energies = result

    # Then
    print(f"states: {states}")
    print(f"energies: {energies}")
    assert len(states) == num_samples * num_vars
    assert len(energies) == num_samples
