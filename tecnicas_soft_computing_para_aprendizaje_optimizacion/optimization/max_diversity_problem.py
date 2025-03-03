import random
from typing import List, Tuple

import numpy as np
import pandas as pd


def greedy_heuristic(n: np.ndarray, m: int) -> List[int]:
    """
    Implements a greedy solution for the Maximum Diversity Problem.

    Args:
    n: 1D NumPy array containing the elements.
    m: Number of elements to select.

    Returns:
    List[int]: Indices of the selected elements.
    """
    if m > len(n):
        raise ValueError("m cannot be larger than the total number of elements in n")

    # Start with the most diverse pair (max and min value indices)
    max_ind, min_ind = int(np.argmax(n)), int(np.argmin(n))
    solution = [max_ind, min_ind]

    # Iteratively add elements that maximize diversity
    while len(solution) < m:
        remaining = [i for i in range(len(n)) if i not in solution]
        best_next = max(
            remaining, key=lambda i: sum(abs(n[i] - n[j]) for j in solution)
        )
        solution.append(best_next)

    return solution


def local_search(n: np.ndarray, m: int, max_iterations: int = 100) -> List[int]:
    """
    Implements a local search for the Maximum Diversity Problem.
    Args:
        n: 1D NumPy array containing the elements.
        m: Number of elements to select.
        max_iterations: Maximum number of iterations for local search.
    Returns:
        List[int]: Indices of the selected elements.
    """
    if m > len(n):
        raise ValueError("m cannot be larger than the total number of elements in n")

    # Start with a greedy solution as the initial solution
    solution = greedy_heuristic(n, m)

    # Evaluate the current solution
    current_value = evaluate_solution(solution, n)

    iteration = 0
    improved = True

    while iteration < max_iterations and improved:
        improved = False

        # Generate all possible swaps and find the best one
        best_swap = None
        best_swap_value = current_value

        for i in range(len(solution)):
            # Consider all elements not in the solution
            for j in range(len(n)):
                if j not in solution:
                    # Create a new solution by swapping
                    new_solution = solution.copy()
                    new_solution[i] = j

                    # Evaluate the new solution
                    new_value = evaluate_solution(new_solution, n)

                    # Update best swap if this one is better
                    if new_value > best_swap_value:
                        best_swap_value = new_value
                        best_swap = (i, j)
                        improved = True

        # Apply the best swap if one was found
        if improved:
            i, j = best_swap
            solution[i] = j
            current_value = best_swap_value

        iteration += 1

    return solution


def evaluate_solution(solution: List[Tuple], n: np.ndarray) -> float:
    return sum(np.abs(n[idx]) for idx in solution)


def run_diversity_experiment(
    num_experiments: int = 30,
    n_size: int = 1000,
    value_range: Tuple = (-2000, 2000),
    m: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs experiments comparing greedy, local search, and random solutions for the
    Maximum Diversity Problem and reports the results.

    Args:
        num_experiments: Number of experiments to run
        n_size: Number of random elements to generate
        value_range: Range of random values (min, max)
        m: Number of elements to select
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Results of all experiments
    """
    # Set random seeds for reproducibility
    #np.random.seed(seed)
    #random.seed(seed)

    # Create a DataFrame to store results
    results = []

    for exp in range(num_experiments):
        # Generate random data
        n = np.random.uniform(value_range[0], value_range[1], size=n_size)

        # Generate solutions using each method
        greedy_solution = greedy_heuristic(n, m)
        local_search_solution = local_search(n, m)
        random_solution = random.sample(range(n_size), m)

        # Evaluate solutions
        greedy_value = evaluate_solution(greedy_solution, n)
        local_search_value = evaluate_solution(local_search_solution, n)
        random_value = evaluate_solution(random_solution, n)

        # Calculate improvement ratios
        greedy_over_random = (
            greedy_value / random_value if random_value > 0 else float("inf")
        )
        local_over_random = (
            local_search_value / random_value if random_value > 0 else float("inf")
        )
        local_over_greedy = (
            local_search_value / greedy_value if greedy_value > 0 else float("inf")
        )

        # Store results
        results.append(
            {
                "experiment": exp + 1,
                "greedy_value": greedy_value,
                "local_search_value": local_search_value,
                "random_value": random_value,
                "greedy_over_random": greedy_over_random,
                "local_over_random": local_over_random,
                "local_over_greedy": local_over_greedy,
            }
        )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Display summary statistics
    print("=== MAXIMUM DIVERSITY PROBLEM EXPERIMENT RESULTS ===")
    print(f"Parameters: {num_experiments} experiments, n_size={n_size}, m={m}")
    print("\nAVERAGE DIVERSITY VALUES:")

    for method in ["greedy", "local_search", "random"]:
        mean_val = df_results[f"{method}_value"].mean()
        std_val = df_results[f"{method}_value"].std()
        print(f"{method.upper()} solution: {mean_val:.2f} ± {std_val:.2f}")

    print("\nIMPROVEMENT RATIOS:")
    print(f"Greedy over Random: {df_results['greedy_over_random'].mean():.2f}x")
    print(f"Local Search over Random: {df_results['local_over_random'].mean():.2f}x")
    print(f"Local Search over Greedy: {df_results['local_over_greedy'].mean():.2f}x")

    return df_results


if __name__ == "__main__":
    results = run_diversity_experiment()
