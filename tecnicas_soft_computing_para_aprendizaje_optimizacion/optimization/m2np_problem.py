import random
from typing import Tuple

import numpy as np
import pandas as pd


def greedy_heuristic(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a greedy solution for the M2NP problem.
    Args:
        s: 2D NumPy array representing a collection of vectors.
    Returns:
        Tuple[np.ndarray, np.ndarray] Tuple of 2D arrays, each containing a collection of vectors.
    """

    # Get number of vectors and dimensions
    _, d = s.shape

    # Initialize sets S1 and S2 with indices
    s1_indices = []
    s2_indices = []

    # Initialize sums for each set
    s1_sum = np.zeros(d)
    s2_sum = np.zeros(d)

    # Sort vectors by norm (as a heuristic to process more impactful vectors first)
    vector_norms = np.linalg.norm(s, axis=1)
    sorted_indices = np.argsort(vector_norms)[::-1]  # Sort in descending order

    # Greedily assign each vector to minimize the maximum difference
    for idx in sorted_indices:
        vector = s[idx]

        # Calculate potential sums if the vector is added to S1 or S2
        potential_s1_sum = s1_sum + vector
        potential_s2_sum = s2_sum + vector

        # Calculate the maximum difference for both potential assignments
        max_diff_if_added_to_s1 = np.max(np.abs(potential_s1_sum - s2_sum))
        max_diff_if_added_to_s2 = np.max(np.abs(s1_sum - potential_s2_sum))

        # Assign to the set that minimizes the maximum difference
        if max_diff_if_added_to_s1 <= max_diff_if_added_to_s2:
            s1_indices.append(idx)
            s1_sum = potential_s1_sum
        else:
            s2_indices.append(idx)
            s2_sum = potential_s2_sum

    # Create the final sets of vectors
    s1 = s[s1_indices]
    s2 = s[s2_indices]

    return s1, s2


def local_search(
    s: np.ndarray, max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a local search for the Maximum Diversity Problem.

    Args:
        s (np.ndarray): 2D NumPy array representing a collection of vectors.
        max_iterations (int): Maximum number of iterations for local search.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of 2D arrays (s1, s2), representing two subsets of s.
    """
    solution = greedy_heuristic(s)
    current_value = evaluate_solution(solution)

    iteration = 0
    improved = True

    while iteration < max_iterations and improved:
        improved = False
        best_solution = solution
        best_value = current_value

        s1, s2 = solution

        # Try moving elements from s1 to s2
        for i in range(s1.shape[0]):
            new_s1 = np.delete(s1, i, axis=0)
            element = s1[i]
            new_s2 = np.vstack([s2, element]) if s2.size else np.array([element])

            # Ensure s1 is not empty
            if new_s1.shape[0] == 0:
                continue

            new_solution = (new_s1, new_s2)
            new_value = evaluate_solution(new_solution)

            if new_value < best_value:
                best_value = new_value
                best_solution = new_solution
                improved = True

        # Try moving elements from s2 to s1
        for j in range(s2.shape[0]):
            new_s2 = np.delete(s2, j, axis=0)
            element = s2[j]
            new_s1 = np.vstack([s1, element]) if s1.size else np.array([element])

            # Ensure s2 is not empty
            if new_s2.shape[0] == 0:
                continue

            new_solution = (new_s1, new_s2)
            new_value = evaluate_solution(new_solution)

            if new_value < best_value:
                best_value = new_value
                best_solution = new_solution
                improved = True

        # Apply the best swap found
        if improved:
            solution = best_solution
            current_value = best_value

        iteration += 1

    return solution


def evaluate_solution(solution: Tuple[np.ndarray, np.ndarray]) -> float:
    sum_s1 = np.sum(solution[0], axis=0)
    sum_s2 = np.sum(solution[1], axis=0)
    return np.max(np.abs(sum_s1 - sum_s2))


def run_experiment(
    num_experiments: int = 30,
    size: Tuple[int] = (30, 1000),
    value_range: Tuple = (1, 10),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs experiments comparing greedy, local search, and random solutions for the
    M2NP Problem and reports the results.

    Args:
        num_experiments: Number of experiments to run
        size: Size of the original set of vectors
        value_range: Range of random values (min, max)
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Results of all experiments
    """
    # Set random seeds for reproducibility
    # np.random.seed(seed)
    # random.seed(seed)

    # Create a DataFrame to store results
    results = []

    for exp in range(num_experiments):
        # Generate random data
        s = generate_vectors(n=size[0], d=size[1], value_range=value_range)

        # Generate solutions using each method
        greedy_solution = greedy_heuristic(s)
        local_search_solution = local_search(s)
        random_solution = generate_random_solution(s)

        # Evaluate solutions
        greedy_value = evaluate_solution(greedy_solution)
        local_search_value = evaluate_solution(local_search_solution)
        random_value = evaluate_solution(random_solution)

        # Calculate improvement differences (for minimization, positive means better)
        greedy_vs_random = (
            random_value - greedy_value
        )  # Positive means greedy is better than random
        local_vs_random = (
            random_value - local_search_value
        )  # Positive means local is better than random
        local_vs_greedy = (
            greedy_value - local_search_value
        )  # Positive means local is better than greedy

        # Store results
        results.append(
            {
                "experiment": exp + 1,
                "greedy_value": greedy_value,
                "local_search_value": local_search_value,
                "random_value": random_value,
                "greedy_over_random": greedy_vs_random,
                "local_over_random": local_vs_random,
                "local_over_greedy": local_vs_greedy,
            }
        )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Display summary statistics
    print("=== M2NP PROBLEM EXPERIMENT RESULTS ===")
    print(f"Parameters: {num_experiments} experiments, size={size}")
    print("\nAVERAGE FITNESS VALUES (lower is better):")
    for method in ["greedy", "local_search", "random"]:
        mean_val = df_results[f"{method}_value"].mean()
        std_val = df_results[f"{method}_value"].std()
        print(f"{method.upper()} solution: {mean_val:.2f} ± {std_val:.2f}")

    print("\nABSOLUTE IMPROVEMENTS (positive = better):")
    print(f"Greedy over Random: {df_results['greedy_over_random'].mean():.2f}")
    print(f"Local Search over Random: {df_results['local_over_random'].mean():.2f}")
    print(f"Local Search over Greedy: {df_results['local_over_greedy'].mean():.2f}")

    return df_results


def generate_vectors(n, d, value_range=(1, 10)):
    """
    Generates an array of n vectors of dimension d.

    :param n: Number of vectors
    :param d: Dimension of each vector
    :param value_range: Tuple (min, max) for the generated values (default is between 0 and 1)
    :return: NumPy array of shape (n, d)
    """
    return np.random.uniform(value_range[0], value_range[1], (n, d))


def generate_random_solution(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates random solution for the M2NP problem.

    Args:
    s: 2D NumPy array representing a collection of vector.

    Returns:
    Tuple[np.ndarray, np.ndarray] Tuple of 2D arrays, each containing a collection of vectors.
    """
    s1_indexes = np.random.default_rng().choice(
        range(s.shape[0]), size=random.randint(1, s.shape[0] - 1), replace=False
    )

    s2_indexes = np.where(~np.isin(list(range(0, s.shape[0])), s1_indexes))[0]

    s1 = s[s1_indexes]
    s2 = s[s2_indexes]
    return s1, s2


if __name__ == "__main__":
    results = run_experiment()
