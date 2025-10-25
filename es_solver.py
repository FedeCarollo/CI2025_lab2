import numpy as np
from solution import Solution
from copy import deepcopy

class TSPESSolver:
    """
    Evolutionary strategy solution for solving TSP problem
    """

    def __init__(self, cost_matrix, population_size: int = 100, offspring_size: int = 50, mutation_rate: float = 0.01, greedy_initial_solutions: bool = True, tau: int=20):
        self.cost_matrix = cost_matrix
        self.solution_size = cost_matrix.shape[0]
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.cost_simmetry = (cost_matrix.T == cost_matrix).all()
        self.tau = tau

        if greedy_initial_solutions:
            self.population = [self._make_greedy_solution(np.random.rand()) for _ in range(self.population_size)]
        else:
            self.population = [self._make_random_solution() for _ in range(self.population_size)]
        self.population.sort(key=lambda sol: sol.fitness)   #Keep population sorted by fitness

    def _make_random_solution(self):
        sequence = np.random.permutation(self.solution_size)
        return self._new_solution(sequence)
        

    def _make_greedy_solution(self, eps: float = 0.1):
        """
        Generate greedy solution, with probability eps choose instead a random destination
        """
        solution = -np.ones(self.solution_size, dtype=int)
        initial = np.random.randint(0, self.solution_size-1)

        solution[0] = initial


        for pos in range(1, self.solution_size):
            cur = int(solution[pos-1])
            row = self.cost_matrix[cur, :].copy()
            visited = solution[:pos].astype(int)
            row[visited] = np.inf

            if np.random.rand() < eps:
                # Choose randomly
                choices = np.where(row < np.inf)[0]
                next_step = int(np.random.choice(choices))
            else:
                # Choose among lowest next stop costs
                choices = choices = np.where(row == row.min())[0]
                # If multiple minimums, then choose a random one
                next_step = int(np.random.choice(choices))
            solution[pos] = next_step
        return self._new_solution(solution)
    
    def _new_solution(self, sequence: np.ndarray):
        return Solution(sequence, self._compute_fitness(sequence))
    
    def _compute_fitness(self, sequence: np.ndarray) -> float:
        seq = np.asarray(sequence, dtype=int)
        if seq.size == 0:
            return 0.0
        
        next_idx = np.roll(seq, -1)
        return float(np.sum(self.cost_matrix[seq, next_idx]))
    
    def _selection(self, tau: int = 5) -> list[list[Solution]]:
        """
        Select parents using tournament selection
        """
        pop_n = len(self.population)
        if tau > pop_n:
            tau = pop_n

        selected = []
        for _ in range(self.offspring_size):
            current = []
            for _ in range(2):  #Select two parents
                competitors = np.random.choice(self.population, size=tau, replace=False)
                winner = min(competitors, key=lambda sol: sol.fitness)
                current.append(winner)
            selected.append(current)
        return selected
    
    def _selection_better(self, tau: int = 5) -> list[list[Solution]]:
        """
        Select parents using tournament selection (numpy optimized)
        """
        pop_n = len(self.population)
        if tau > pop_n:
            tau = pop_n

        fitness = np.array([sol.fitness for sol in self.population])
        indices = np.arange(pop_n)

        # Select tau competitors for each parent of each offspring
        competitors = np.stack([
            np.random.choice(indices, size=(self.offspring_size, tau), replace=True)
            for _ in range(2)  #Select two parents
        ], axis=1)  # Shape: (offspring_size, 2, tau)

        # Compute fitness of each competitor
        competitors_fitness = fitness[competitors]  # Shape: (offspring_size, 2, tau)

        # Select the best competitor for each parent of each offspring
        winners_indices = np.argmin(competitors_fitness, axis=2)  # Shape: (offspring_size, 2)


        offspring_idx = np.arange(self.offspring_size)[:, np.newaxis]       # Shape: (offspring_size, 1)
        parent_idx = np.arange(2)[np.newaxis, :]                            # Shape: (1, 2)
        winners = competitors[offspring_idx, parent_idx, winners_indices]   # Shape: (offspring_size, 2)

        selected = [
            [self.population[winner] for winner in pair]
            for pair in winners
        ]
        return selected



    
    def _crossover(self, p1: Solution, p2: Solution) -> Solution:
        """
        Pick 2 crossover points and exchange the segments between parents
        Then fill the remaining cities from the other parent in order
        """

        cx1, cx2 = np.sort(np.random.choice(self.solution_size, size=2, replace=False))

        child_sequence = -np.ones(self.solution_size, dtype=int)
        # Copy segment from first parent
        child_sequence[cx1:cx2] = p1.sequence[cx1:cx2]

        child_cities = set(child_sequence[cx1:cx2].tolist())

        # Fill remaining cities from second parent
        current_pos = cx2 % self.solution_size
        for city in p2.sequence:
            if city not in child_cities:
                child_sequence[current_pos] = city
                child_cities.add(city)
                current_pos = (current_pos + 1) % self.solution_size
        return self._new_solution(child_sequence)
    
    def _mutate(self, solution: Solution) -> Solution:
        """
        Mutate solution by swapping two cities with probability mutation_rate
        """
        new_sequence = solution.sequence.copy()
        for i in range(self.solution_size):
            if np.random.rand() < self.mutation_rate:
                j = np.random.randint(0, self.solution_size)
                new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
        return self._new_solution(new_sequence)
                

    def es_solve(self, max_iter: int = 1000):
        """
        Evolution Strategy algorithm to solve TSP
        """
        best_solution = min(self.population, key=lambda sol: sol.fitness)

        history = [best_solution.fitness]
        best_history = [best_solution.fitness]

        for iteration in range(max_iter):
            #Get parents for offspring generation
            parents = self._selection_better(tau=self.tau)
            #Generate offspring from combining parents
            offspring = [self._crossover(p1, p2) for p1, p2 in parents]
            #Mutate some offspring
            final_offspring = [self._mutate(o) if np.random.rand() < self.mutation_rate else o for o in offspring]
            # Select next generation
            combined = self.population + final_offspring
            # combined = final_offspring
            combined.sort(key=lambda sol: sol.fitness)
            # Keep best POP_SIZE individuals
            self.population = combined[:self.population_size]

            if self.population[0].fitness < best_solution.fitness:
                best_solution = deepcopy(self.population[0])

            history.append(self.population[0].fitness)
            best_history.append(best_solution.fitness)


        return best_solution, best_history, history
