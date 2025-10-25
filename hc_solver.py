import numpy as np
from solution import Solution
from copy import deepcopy

class TSPHCSolver:
    """
    Hill climber solution for solving TSP problem
    """

    def __init__(self, cost_matrix, sol: np.ndarray = None):
        self.cost_matrix = cost_matrix
        self.solution_size = cost_matrix.shape[0]
        if sol is None:
            sol = self._make_greedy_solution()
        self.sol = sol
        

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
                next_step = int(np.random.choice(choices))
            solution[pos] = next_step
            solution = np.array(solution)
        return self._new_solution(solution)
    
    def _new_solution(self, sequence: np.ndarray):
        return Solution(sequence, self._compute_fitness(sequence))
    
    def _compute_fitness(self, sequence: np.ndarray) -> float:
        cost = 0
        from_loc = sequence[0]
        for i in range(sequence.shape[0]-1):
            to_loc = sequence[i+1]
            cost += self.cost_matrix[from_loc, to_loc]
            from_loc = to_loc
        cost+=self.cost_matrix[from_loc, sequence[0]]   #Count cycle closure
        return cost
    
    def hc_solve(self, max_iter: int = 1000):
        """
        Hill climbing algorithm to solve TSP
        """
        current_solution = deepcopy(self.sol)
        best_solution = deepcopy(current_solution)

        current_history = [current_solution.fitness]
        best_history = [best_solution.fitness]

        for iteration in range(max_iter):
            # Tweak solution
            i,j = np.random.choice(self.solution_size, size=2, replace=False)
            new_sequence = current_solution.sequence.copy()
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
            new_solution = self._new_solution(new_sequence)
            # If new solution is better, keep it
            if new_solution.fitness <= current_solution.fitness:
                current_solution = new_solution
                # Update best solution found
                if current_solution.fitness < best_solution.fitness:
                    best_solution = current_solution
            elif np.log(np.random.random()) < (current_solution.fitness - new_solution.fitness) / (0.1*(max_iter - iteration)/max_iter):
                current_solution = new_solution
            # Store history
            current_history.append(current_solution.fitness)
            best_history.append(best_solution.fitness)
        self.best_solution = deepcopy(best_solution)

        return best_solution, current_history, best_history

