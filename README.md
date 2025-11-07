# Traveling Salesman Problem

### Initial considerations

I noticed that the notebook became too dense and with too much code, so the notebook really just calls python modules where the real code is.

Please be sure to check out them!

### Feasibility considerations

I couldn't figure out an efficient way to accept unfeasible solutions and work towards a feasible solution, so my initial assumption was to always go from a feasible solution to another, without accepting any intermediary unfeasible solutions.

## Solution representation

### Sequence representation

Each solution presented is represented as a sequence of $N$ indices in range $[0, \textrm{N-1}]$ where $N$ is the number of nodes.

e.g. $[0, 4, 1, 2, 3]$ for $N = 5$

### Fitness representation

Since I did not accept unfeasible solutions, the natural way to define fitness was as the sum of costs along the sequence.

The objective is then to minimize such fitness.

### Solution class

Each proposed solution is represented with a `Solution` object class with fields `sequence` and `fitness`

You can find the `Solution` dataclass in [solution.py](solution.py)

## First approach with hill climber

At first, as suggested the the professor, I approached the problem with a simple hill climber approach, including simulated annealing and initial greedy solution

### Simulated annealing

Simulated annealing allows to accept a non optimal solution with probability

$$
    p = e^{-\frac{(f_n - f_o)}{T}}
$$

Where

- $f_n$ is the fitness of the current solution
- $f_o$ is the fitness of the old solution
- $T$ is the temperature

I decided to use a temperature schedule as a function of current iteration

$$
T_k = 0.1 \cdot \frac{\text{MAX ITER} - \text{CUR ITER}}{\text{MAX ITER}}
$$

To encourage initial exploration

### Greedy initialization

I figured that starting from already "optimal" solutions would have improved drastically the final best solution.
Just to mantain some variance in sight of future evolution approach, I introduced also a random component into the algorithm as explained later.
The pseudocode for a greedy solution is

```
sequence = [random_starting_index]

for i in [1, N]
    with probability p
        idx = 'select next index that minimizes cost from last node and not already selected'
    else
        idx = 'select next index randomly among those not already chosen'

    sequence.add(idx)
return sequence
```

I did some testing using instead random permutations and, as expected, the greedy_init approach outperformed random_init by far, also when given 10 times less iterations.

### Why Hill Climber

Hill Climber is fast and gives good solutions.

I mainly used it as a benchmark agains ES strategies, to ensure the latter performed better (it was not always the case).

You can find the hill climber solution in [hc_solver.py](hc_solver.py)

## ES strategy

After implementing the hill climber, I proceeded to implement an Evolutionary Strategy (ES) approach to solve the TSP problem.

The adopted strategy is a ($\mu+\lambda$), where $\lambda$ represents the offspring size (offspring_size) and $\mu$ the population size (population_size).

At each generation, $\lambda$ individuals are generated from the current population, and then the best $\mu$ individuals are selected from the previous population and the new offspring.

I also tried a ($\mu, \lambda$) apporach but it consistently peformed worse in all instances so I decided to discard it.

### Initialization parameters

The main configurable parameters are:

- **population_size** ($\mu$): number of individuals in the population
- **offspring_size** ($\lambda$): number of individuals generated per generation
- **mutation_rate**: probability of mutation per gene
- **greedy_initial_solutions**: whether to initialize the population with greedy or random solutions
- **tau** ($\tau$): tournament size for selection

### Selection

Parent selection is performed using tournament selection with tournament size tau.

For each required parent pair, $\tau$ individuals are randomly selected from the population, and the winner is the one with the best fitness.

I implemented two versions of the selection function:

- A "pythonic" version (`_selection`) using loops and lists (not used, I kept it because it is easie to understand)
- A NumPy-optimized version (`_selection_better`) leveraging vectorized operations for better performance, especially with large populations

### Crossover

Crossover selects two random cut points in the parent sequences. The segment between these points is copied from the first parent, and the remaining nodes are filled from the order of the second parent, maintaining feasibility.

### Mutation

The approach for selection was to select two parents for each offspring, so it didn't make sense to perform either mutation or crossover. Crossover has to occur always, then each offspring can randomly be mutated.

I introduced mutation for each offspring with probability `mutation_rate`

Mutation occurs with probability `mutation_rate` per position in the sequence: if triggered, two random positions are swapped.

You can find the complete ES strategy implementation in [es_solver.py](es_solver.py)

## Strategy testing

### Reproducibility

For reproducibility of the results I used `np.random.seed(42)` in accordance to the previous lab.

### Testing framework

The testing framework in [test_solvers.py](test_solvers.py) provides a comprehensive evaluation system that compares HC and ES solvers across multiple TSP instances.

### Test execution

For each problem file in `test_problems/`:

1. **Hill Climber Test** (`_hc_task`): Runs a single HC solver with fixed parameters (10,000 iterations, greedy initialization, simulated annealing)
2. **ES Tests** (`_ec_task`): Runs multiple ES configurations in **parallel** using `joblib.Parallel` with all available CPU cores

Parallelization is not the scope of this course so just believe it speeds up computations if you don't understand the code.

### Parameter combinations

Two preset options are available:

- `easy_combinations()`: 32 configurations (2 mutations × 2 populations × 2 offsprings × 2 greedy options)
- `default_combinations()`: 108 configurations (3 mutations × 3 populations × 3 offsprings × 2 greedy options)

I already provide the solution results for `default_combinations` in `test_problems/results/` so it isn't necessary to run again the benchmark problems.

### Results storage and format

Results are saved in `test_problems/results/` with naming pattern: `problem_{name}_{solver}_results.npy`

Each result is a `SolutionResults` object class containing:

- `best_solution`: Solution object with optimal sequence and fitness
- `best_history` & `history`: Fitness evolution (compressed using Run-Length Encoding)
- `best_fitness`: Final best fitness value
- `best_sequence`: Best tour as list of indices
- `params`: Dictionary with solver parameters used

**Compression**: Fitness histories are compressed using RLE encoding. For example: `[100, 100, 100, 95, 95, 90]` becomes `{'values': [100, 95, 90], 'counts': [3, 2, 1]}`. This saves **a lot** of storage space, especially useful when fitness plateaus occur.

### Visualization

The notebook generates 3-panel comparison plots for each problem using `get_results()`:

1. **Upper panel - Hill Climber**: Current solution vs best solution over iterations
2. **Left panel - ES (greedy init)**: Overlayed results for all greedy-initialized ES configurations
3. **Right panel - ES (random init)**: Overlayed results for all random-initialized ES configurations

Each overlay shows the configuration parameters (mutation rate, population size, offspring size) to identify which parameter combinations perform best.

### Final results CSV

The file `tsp_best_solutions.csv` contains the best solutions found across all solvers and parameter combinations for each problem (using `seed=42` for reproducibility):

```
problem_name,best_fitness,best_sequence
g_10,1497.66,[7, 9, 5, 4, 6, 1, 3, 2, 8, 0]
r1_1000,2802.24,[120, 49, 129, 960, ...]
r2_1000,-49201.33,[978, 71, 445, ...]
```

For each problem, the system compares:

- Best ES result (minimum fitness across all parameter combinations)
- HC result
- Writes the overall best to the CSV

Values can be positive (standard TSP distances) or negative (special problem variants).

### Parameters checking

After I found the best parameter configuration for each problem, I ran `n_runs=10` simulations for each model using the best parameters to ensure the performance, evaluating mean, variance and percentile

- **In 13/21 problems (62%)** the solution initially found is better or comparable to the next 10 runs confirming that parameters are good
- **In 8/21 problems (38%)** new runs perform even better
  - Parameters are stable and generate good solutions
  - Solutions can be improved by running more iterations, as I expected
  - Varianza is well handled

There are few extreme percentiles (high and low), meaning the parameters are consistent and representatives, not fortunate outliers.

I also included the new best_solutions found and associated fitness, together with the fitnesses of each run.

The best fitnesses are reported here in tabular form for simplicity

| Problem | Best Fitness       |
| ------- | ------------------ |
| g_10    | 1497.6636482252907 |
| g_20    | 1755.5146770830047 |
| g_50    | 2834.9558213652563 |
| g_100   | 4274.537166515126  |
| g_200   | 6351.460141727046  |
| g_500   | 9985.053842119225  |
| g_1000  | 14516.982187947051 |
| r1_10   | 184.27344079993895 |
| r1_20   | 342.42188065666977 |
| r1_50   | 588.3529516509842  |
| r1_100  | 762.1211140896534  |
| r1_200  | 1135.511816727526  |
| r1_500  | 1758.8645142877922 |
| r1_1000 | 2772.0944330327957 |
| r2_10   | -411.7017155524985 |
| r2_20   | -844.2659774812848 |
| r2_50   | -2274.489282358286 |
| r2_100  | -4712.857397154966 |
| r2_200  | -9595.193023920743 |
| r2_500  | -24518.59064685426 |
| r2_1000 | -49362.42122643598 |

**Note**: please note that overall best only includes the tests ran on the 10 instances of each problem, not accounting the original best found in the previous point, I forgot to add it, so if you compare the table and the file please pick the best among `stored_best` and `overall_best` (I thought it unnecessary to run the simulation again just to add that detail).

You can find the summary results in [tsp_best_tuned.json](tsp_best_tuned.json)

## Conclusions

### Performance comparison

The testing results reveal interesting trade-offs between the two approaches:

**Hill Climber**: Extremely fast and provides good solutions efficiently. With 10,000 iterations, it solves problems quickly and is ideal as a baseline benchmark for evaluating ES performance.

**Evolutionary Strategy**: Computationally more expensive but generally produces better solutions than HC, especially with careful parameter tuning. The population-based approach allows exploration of multiple search directions simultaneously, leading to more robust performance across diverse problem instances.

### Quality of solutions

A key observation is that evolutionary solutions improve only marginally over greedy initialization. This is likely because:

- Greedy initialization already produces solutions very close to local optima
- The search space contains many local optima, making significant improvement difficult
- Random initialization, conversely, benefits substantially more from the evolutionary process, showing the value of ES when good initialization is unavailable

This suggests that greedy solutions are exceptionally strong starting points, and the evolutionary strategy's main advantage lies in its consistency and robustness rather than dramatic improvements over a single well-initialized solution.

### Best solutions

By running multiple instances of the same problems I found better results than those reported in `tsp_best_solutions.csv` as I explained before, so keep the table reported as the ground truth for the best fitnesses for each problem.

### Computational trade-offs

While better solutions could theoretically be obtained through:

- Increased iterations per solver
- Finer-grained parameter grids
- More extensive problem coverage
- Hybrid approaches combining HC and ES

The computational cost becomes prohibitive when scaling across many problems and parameter combinations.

The current testing framework represents a practical balance between solution quality and computational feasibility on available hardware.

### Testing and experimentation

The notebook contains dedicated cells at the bottom for independent testing of any solver configuration. Users can:

- Load individual problems and test custom solvers
- Configure arbitrary parameter sets
- Run focused single experiments with immediate feedback
- Visualize results in real-time without re-running the full benchmark suite

## Collaborations

I shared some ideas at the start of the laboratory with

- Davide Carletto s339425
- Alessandro Benvenuti s343748

But we developed code independently

I used AI models mainly to help me with plots and parallelization of the work and also to make "numpy efficient" some functions that I already wrote to drastically improve performance (up to 100x)

Main ideas come from me by testing and tweaking parameters to improve solutions.
