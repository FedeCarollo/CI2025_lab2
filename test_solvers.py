from solution import Solution
import numpy as np
from hc_solver import TSPHCSolver
from es_solver import TSPESSolver


def rle_encode(arr):
    """
    Encode array using Run-Length Encoding compression.
    Returns dict with 'values' and 'counts' for compressed representation.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return {'values': [], 'counts': []}
    
    # Find where values change
    change_indices = np.where(np.diff(arr) != 0)[0] + 1
    values = np.split(arr, change_indices)
    counts = [len(v) for v in values]
    values = [v[0] for v in values]
    
    return {
        'values': [float(v) for v in values],
        'counts': [int(c) for c in counts]
    }


def rle_decode(compressed):
    """
    Decode Run-Length Encoded data back to array.
    """
    if not compressed['values']:
        return []
    
    result = []
    for value, count in zip(compressed['values'], compressed['counts']):
        result.extend([value] * count)
    
    return result


class SolutionResults:
    def __init__(self, best_solution: Solution, best_history: list[float], history: list[float], 
                 best_fitness: float, best_sequence: list, params: dict):
        self.best_solution = best_solution
        self.best_history = best_history
        self.history = history
        self.best_fitness = best_fitness
        self.best_sequence = best_sequence
        self.params = params

def _ec_task(current_problem, mutation, population, offspring, use_greedy):
    np.random.seed(42)
    ec_solver = TSPESSolver(current_problem, 
                            population_size=population, offspring_size=offspring, 
                            mutation_rate=mutation, greedy_initial_solutions=use_greedy)
    ec_best_sol, ec_best_history, ec_history = ec_solver.es_solve(max_iter=250)

    return SolutionResults(
        best_solution=ec_best_sol,
        best_history=rle_encode(ec_best_history),
        history=rle_encode(ec_history),
        best_fitness=float(ec_best_sol.fitness),
        best_sequence=ec_best_sol.sequence.tolist(),
        params={
            'greedy_initial': bool(use_greedy),
            'population': int(population),
            'offspring': int(offspring),
            'mutation': float(mutation)
        }
    )

def _hc_task(current_problem):
    np.random.seed(42)
    hc_solver = TSPHCSolver(current_problem)
    hc_best_sol, hc_best_history, hc_history = hc_solver.hc_solve(max_iter=10000)

    return SolutionResults(
        best_solution=hc_best_sol,
        best_history=rle_encode(hc_best_history),
        history=rle_encode(hc_history),
        best_fitness=float(hc_best_sol.fitness),
        best_sequence=hc_best_sol.sequence.tolist(),
        params={'greedy_initial': True}
    )

def easy_combinations() -> list[tuple[float, int, int, bool]]:
    """
    Generate easy parameter combinations for ES solver testing
    """
    mutations = [0.01, 0.1]
    populations = [50, 100]
    offsprings = [25, 50]
    greedy_options = [True, False]

    return [
        (mut, pop, off, greedy)
        for mut in mutations
        for pop in populations
        for off in offsprings
        for greedy in greedy_options
    ]

def default_combinations() -> list[tuple[float, int, int, bool]]:
    """
    Generate grid parameter combinations for ES solver testing
    """
    mutations = [0.01, 0.05, 0.1]
    populations = [50, 100, 200]
    offsprings = [25, 50, 100]
    greedy_options = [True, False]

    return [
        (mut, pop, off, greedy)
        for mut in mutations
        for pop in populations
        for off in offsprings
        for greedy in greedy_options
    ]

import os
from joblib import Parallel, delayed


def execute_test_problems (
    folder: str,
    es_param_combinations: list[tuple[float, int, int, bool]]):

    result_dir = os.path.join(folder, "results")
    os.makedirs(result_dir, exist_ok=True)

    for filename in os.listdir(folder):
        if not filename.endswith('.npy') or not filename.startswith('problem_'):
            continue
        problem_path = os.path.join(folder, filename)
        problem = np.load(problem_path)

        # Hill climbing solver test
        hc_results = _hc_task(problem)

        # Evolutionary strategy solver test
        es_results = Parallel(n_jobs=-1)(
            delayed(_ec_task)(problem, mut, pop, off, greedy)
            for (mut, pop, off, greedy) in es_param_combinations
        )

        # Save results
        base_filename = filename[:-4]  # Remove .npy extension
        hc_output_path = os.path.join(result_dir, f"{base_filename}_hc_results.npy")
        es_output_path = os.path.join(result_dir, f"{base_filename}_es_results.npy")

        np.save(hc_output_path, hc_results)
        np.save(es_output_path, es_results)

def get_results(folder: str, problem_name: str) -> dict:
    result_dir = os.path.join(folder, "results")
    hc_output_path = os.path.join(result_dir, f"{problem_name}_hc_results.npy")
    es_output_path = os.path.join(result_dir, f"{problem_name}_es_results.npy")

    hc_results = np.load(hc_output_path, allow_pickle=True).item()
    es_results = np.load(es_output_path, allow_pickle=True)

    # Decode RLE histories for HC results
    hc_results.best_history = rle_decode(hc_results.best_history)
    hc_results.history = rle_decode(hc_results.history)

    # Decode RLE histories for ES results
    for es_result in es_results:
        es_result.best_history = rle_decode(es_result.best_history)
        es_result.history = rle_decode(es_result.history)

    return {
        'hc_results': hc_results,
        'es_results': es_results
    }
