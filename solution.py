from dataclasses import dataclass
import numpy as np

@dataclass
class Solution:
    sequence: np.ndarray
    fitness: float

def solution_feasible(sol: Solution) -> bool:
    return np.unique(sol.sequence).shape[0] == sol.sequence.shape[0]