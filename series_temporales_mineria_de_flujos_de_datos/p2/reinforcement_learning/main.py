#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CartPole Data Stream Mining Solution

This script implements data stream mining classifiers to learn from expert demonstrations
of the CartPole problem and then replicate the behavior autonomously.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from river import drift, ensemble, metrics, naive_bayes, stream, tree
from tqdm import tqdm

# Constants and configuration
DATA_FILE = "CartPoleInstances.csv"
ENV_NAME = "CartPole-v1"
TEST_EPISODES = 10  # Number of test episodes for performance evaluation


# Feature and target names
def attribute_names():
    return ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]


def class_name():
    return "action"


# Data loading
def load_data_stream():
    """
    Loads the CartPole data stream with proper converters for feature types
    """
    converters = {}
    for attr in attribute_names():
        converters[attr] = float
    converters[class_name()] = lambda x: int(float(x))
    return stream.iter_csv(DATA_FILE, converters=converters, target=class_name())


# Model definitions
def create_models():
    """
    Creates a dictionary of models to compare:
    1. Naive Bayes (baseline)
    2. Hoeffding Adaptive Tree (built-in drift adaptation)
    3. ADWIN-wrapped Logistic Regression (explicit drift detection)
    """
    # Baseline model: Naive Bayes with Gaussian distribution
    nb_model = naive_bayes.GaussianNB()

    # Hoeffding Adaptive Tree (has drift detection built-in)
    ht_model = tree.HoeffdingTreeClassifier()

    # Logistic Regression with ADWIN wrapper for drift detection
    nb_adwin_model = ensemble.ADWINBoostingClassifier(
        model=naive_bayes.GaussianNB(), n_models=10
    )

    return {
        "NaiveBayes": nb_model,
        "HoeffdingTreeClassifier": ht_model,
        "NaiveBayes_ADWIN": nb_adwin_model,
    }


# Function to explicitly check for drift using ADWIN
def detect_drift_with_adwin(data_stream):
    """
    Analyzes the data stream for concept drift using ADWIN detector
    """
    print("Analyzing data stream for concept drift using ADWIN...")

    # Create ADWIN detectors for each feature
    detectors = {
        "Cart Position": drift.ADWIN(delta=0.002),
        "Cart Velocity": drift.ADWIN(delta=0.002),
        "Pole Angle": drift.ADWIN(delta=0.002),
        "Pole Angular Velocity": drift.ADWIN(delta=0.002),
        "action": drift.ADWIN(delta=0.002),
    }

    # Counters for drift detection
    drift_points = {name: [] for name in detectors.keys()}
    total_drifts = {name: 0 for name in detectors.keys()}

    # Process the stream
    for i, (x, y) in enumerate(tqdm(data_stream)):
        # Check each feature for drift
        for feature, value in x.items():
            if detectors[feature].update(value):
                total_drifts[feature] += 1
                drift_points[feature].append(i)

        # Check target variable for drift
        if detectors["action"].update(y):
            total_drifts["action"] += 1
            drift_points["action"].append(i)

    # Report results
    print("\nADWIN Drift Detection Results:")
    for feature, count in total_drifts.items():
        print(f"{feature}: {count} drift points detected")
        if count > 0:
            print(f"  First few drift points: {drift_points[feature][:5]}")

    return drift_points, total_drifts


# Training function
def train_models(models, data_stream):
    """
    Trains all models on the data stream and tracks accuracy metrics
    """
    print("Training models on data stream...")

    # Metrics to track
    accuracy_metrics = {name: metrics.Accuracy() for name in models.keys()}

    # For concept drift analysis
    position_values = []
    velocity_values = []
    angle_values = []
    angular_velocity_values = []
    action_values = []

    # Track accuracy over time
    accuracy_over_time = {name: [] for name in models.keys()}
    time_points = []

    # Train each model on the stream
    for i, (x, y) in enumerate(tqdm(data_stream)):
        # Store values for drift analysis (sample to avoid excessive memory use)
        if i % 100 == 0:
            position_values.append(x["Cart Position"])
            velocity_values.append(x["Cart Velocity"])
            angle_values.append(x["Pole Angle"])
            angular_velocity_values.append(x["Pole Angular Velocity"])
            action_values.append(y)

        # Train each model and update metrics
        for name, model in models.items():
            # Make prediction (if the model has seen enough examples)
            if i > 0:
                y_pred = model.predict_one(x)
                accuracy_metrics[name].update(y, y_pred)

            # Train the model
            model.learn_one(x, y)

        # Record accuracy at regular intervals
        if i % 1000 == 0 and i > 0:
            time_points.append(i)
            for name, metric in accuracy_metrics.items():
                accuracy_over_time[name].append(metric.get())

    # Get final accuracy scores
    accuracy_scores = {name: metric.get() for name, metric in accuracy_metrics.items()}
    print("Training completed. Final accuracy scores:")
    for name, score in accuracy_scores.items():
        print(f"{name}: {score:.4f}")

    # Plot accuracy over time
    plt.figure(figsize=(10, 6))
    for name, values in accuracy_over_time.items():
        plt.plot(time_points, values, label=name)

    plt.title("Model Accuracy Over Time")
    plt.xlabel("Instances Processed")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/accuracy_over_time.png")

    # Return data for further analysis
    drift_data = {
        "position": position_values,
        "velocity": velocity_values,
        "angle": angle_values,
        "angular_velocity": angular_velocity_values,
        "action": action_values,
    }

    return accuracy_scores, drift_data, accuracy_over_time, time_points


# Analyze concept drift from collected data
def analyze_drift(drift_data):
    """
    Analyzes potential concept drift in the data stream from collected statistics
    """
    print("\nAnalyzing potential concept drift in the data stream...")

    # Convert to numpy arrays for easier manipulation
    positions = np.array(drift_data["position"])
    velocities = np.array(drift_data["velocity"])
    angles = np.array(drift_data["angle"])
    angular_velocities = np.array(drift_data["angular_velocity"])
    actions = np.array(drift_data["action"])

    # Create time index (instance number)
    time_index = np.arange(len(positions))

    # Calculate moving averages to visualize trends
    window_size = min(100, len(positions) // 10)

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Plot the moving averages to visualize drift
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(
        time_index[: len(moving_average(positions, window_size))],
        moving_average(positions, window_size),
    )
    plt.title("Cart Position (Moving Average)")
    plt.xlabel("Instance Index")

    plt.subplot(3, 2, 2)
    plt.plot(
        time_index[: len(moving_average(velocities, window_size))],
        moving_average(velocities, window_size),
    )
    plt.title("Cart Velocity (Moving Average)")
    plt.xlabel("Instance Index")

    plt.subplot(3, 2, 3)
    plt.plot(
        time_index[: len(moving_average(angles, window_size))],
        moving_average(angles, window_size),
    )
    plt.title("Pole Angle (Moving Average)")
    plt.xlabel("Instance Index")

    plt.subplot(3, 2, 4)
    plt.plot(
        time_index[: len(moving_average(angular_velocities, window_size))],
        moving_average(angular_velocities, window_size),
    )
    plt.title("Pole Angular Velocity (Moving Average)")
    plt.xlabel("Instance Index")

    plt.subplot(3, 2, 5)
    plt.plot(
        time_index[: len(moving_average(actions, window_size))],
        moving_average(actions, window_size),
    )
    plt.title("Action Distribution (Moving Average)")
    plt.xlabel("Instance Index")
    plt.ylabel("Action (0=Left, 1=Right)")

    plt.tight_layout()
    plt.savefig("img/concept_drift_analysis.png")

    # Check for statistical changes in distribution over time
    # Divide data into segments and compare distributions
    num_segments = 4
    segment_size = len(positions) // num_segments

    print("Analyzing feature distribution changes across time segments:")
    for feature_name, feature_data in [
        ("Cart Position", positions),
        ("Cart Velocity", velocities),
        ("Pole Angle", angles),
        ("Pole Angular Velocity", angular_velocities),
        ("Action", actions),
    ]:
        print(f"\n{feature_name}:")
        segment_means = []
        segment_stds = []

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size
            segment_data = feature_data[start_idx:end_idx]

            mean = np.mean(segment_data)
            std = np.std(segment_data)
            segment_means.append(mean)
            segment_stds.append(std)

            print(f"  Segment {i + 1}: Mean = {mean:.4f}, Std = {std:.4f}")

        # Calculate coefficient of variation of means to check stability
        # Avoid division by zero
        mean_of_means = np.mean(segment_means)
        if mean_of_means != 0:
            mean_cv = np.std(segment_means) / mean_of_means
            print(f"  Coefficient of variation of means: {abs(mean_cv):.4f}")

            if abs(mean_cv) > 0.1:
                print(f"  Potential concept drift detected in {feature_name}")
            else:
                print(f"  No significant drift detected in {feature_name}")
        else:
            print(
                f"  Mean is zero, cannot calculate coefficient of variation for {feature_name}"
            )


# Test model action selection function for evaluation in CartPole
def create_action_selector(model):
    """
    Creates a function that selects actions based on the model's predictions
    """

    def select_action(obs):
        # Convert observation array to dict format for River
        obs_dict = dict(zip(attribute_names(), obs))

        # Get model's prediction
        return model.predict_one(obs_dict)

    return select_action


# Evaluate models in the CartPole environment
def evaluate_model_performance(models):
    """
    Evaluates each model in the CartPole environment and returns performance metrics
    """
    print("\nEvaluating models in the CartPole environment...")

    results = {}

    for name, model in models.items():
        rewards = []
        action_selector = create_action_selector(model)

        for episode in range(TEST_EPISODES):
            env = gym.make(
                ENV_NAME, render_mode=None
            )  # No rendering for faster evaluation
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                # Get action from model
                action = action_selector(obs)

                # Apply action to environment
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
            env.close()

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        results[name] = {"mean": avg_reward, "std": std_reward, "rewards": rewards}
        print(f"{name}: Mean reward = {avg_reward:.2f} ± {std_reward:.2f}")

    # Plot the reward distributions as box plots
    plt.figure(figsize=(10, 6))
    reward_data = [results[name]["rewards"] for name in models.keys()]
    plt.boxplot(reward_data, tick_labels=list(models.keys()))
    plt.title("Reward Distributions Across Models")
    plt.ylabel("Total Reward")
    plt.grid(True, axis="y")
    plt.savefig("img/reward_distributions.png")

    return results


# Visualize a model in action
def visualize_model(model_name, model):
    """
    Visualizes the specified model controlling the CartPole environment
    """
    print(f"\nVisualizing {model_name} in action...")

    action_selector = create_action_selector(model)

    env = gym.make(ENV_NAME, render_mode="human")
    obs, _ = env.reset()
    env.render()

    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        # Get action from model
        action = action_selector(obs)

        # Apply action to environment
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        total_reward += reward

    print(f"Episode ended with reward: {total_reward}")
    print("Press [ENTER] to continue")
    input()
    env.close()


# Statistical comparison of models
def statistical_comparison(accuracy_scores, performance_results):
    """
    Performs statistical comparison between training accuracy and test performance
    """
    print("\nStatistical Comparison of Models:")

    # Create a summary table
    print("\nSummary Table:")
    print("-" * 80)
    print(
        f"{'Model':<25} {'Training Accuracy':<20} {'Test Performance':<20} {'Rank (Acc/Perf)'}"
    )
    print("-" * 80)

    # Get rankings
    acc_sorted = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
    perf_sorted = sorted(
        performance_results.items(), key=lambda x: x[1]["mean"], reverse=True
    )

    acc_ranks = {name: i + 1 for i, (name, _) in enumerate(acc_sorted)}
    perf_ranks = {name: i + 1 for i, (name, _) in enumerate(perf_sorted)}

    for name in accuracy_scores.keys():
        print(
            f"{name:<25} {accuracy_scores[name]:<20.4f} {performance_results[name]['mean']:<20.2f} {acc_ranks[name]} / {perf_ranks[name]}"
        )

    print("-" * 80)

    # Calculate Spearman rank correlation
    acc_values = [accuracy_scores[name] for name in accuracy_scores.keys()]
    perf_values = [
        performance_results[name]["mean"] for name in performance_results.keys()
    ]

    correlation = np.corrcoef(acc_values, perf_values)[0, 1]
    print(f"\nPearson correlation between accuracy and performance: {correlation:.4f}")

    # Calculate rank correlation
    acc_rank_values = [acc_ranks[name] for name in accuracy_scores.keys()]
    perf_rank_values = [perf_ranks[name] for name in performance_results.keys()]
    rank_correlation = np.corrcoef(acc_rank_values, perf_rank_values)[0, 1]
    print(f"Spearman rank correlation: {rank_correlation:.4f}")

    # Interpret results
    print("\nInterpretation:")
    if correlation > 0.7:
        print(
            "Strong positive correlation: Higher training accuracy generally leads to better test performance"
        )
    elif correlation > 0.3:
        print(
            "Moderate positive correlation: Training accuracy somewhat predicts test performance"
        )
    elif correlation > -0.3:
        print(
            "Weak correlation: Training accuracy may not be a reliable predictor of test performance"
        )
    else:
        print(
            "Negative correlation: Higher training accuracy may lead to worse test performance"
        )

    return correlation, rank_correlation


# Main function
def main():
    """
    Main execution function
    """
    # Create models
    models = create_models()

    # First, analyze drift with ADWIN
    print("Step 1: Analyzing data stream for concept drift...")
    data_stream = load_data_stream()
    drift_points, total_drifts = detect_drift_with_adwin(data_stream)

    # Train models on data stream
    print("\nStep 2: Training models on data stream...")
    data_stream = load_data_stream()  # Reload the stream
    accuracy_scores, drift_data, accuracy_over_time, time_points = train_models(
        models, data_stream
    )

    # Analyze drift from collected data
    print("\nStep 3: Analyzing feature distributions for concept drift...")
    analyze_drift(drift_data)

    # Evaluate model performance in CartPole environment
    print("\nStep 4: Evaluating model performance in CartPole environment...")
    performance_results = evaluate_model_performance(models)

    # Statistical comparison
    print("\nStep 5: Comparing training accuracy with test performance...")
    correlation, rank_correlation = statistical_comparison(
        accuracy_scores, performance_results
    )

    # Find best model based on test performance
    best_model_name = max(performance_results.items(), key=lambda x: x[1]["mean"])[0]
    print(f"\nBest performing model: {best_model_name}")

    # Visualize best model
    print("\nStep 6: Visualizing best model in action...")
    visualize_model(best_model_name, models[best_model_name])

    print("\nAnalysis complete.")
    print(f"Best model: {best_model_name}")
    print(f"Training accuracy: {accuracy_scores[best_model_name]:.4f}")
    print(
        f"Test performance: {performance_results[best_model_name]['mean']:.2f} ± {performance_results[best_model_name]['std']:.2f}"
    )

    if correlation > 0.5:
        print(
            "\nTraining accuracy is a good predictor of test performance in this problem."
        )
    else:
        print(
            "\nTraining accuracy is not a reliable predictor of test performance in this problem."
        )

    # Check for concept drift impact
    has_drift = False
    for feature, count in total_drifts.items():
        if count > 0:
            has_drift = True
            break

    if has_drift:
        print("\nConcept drift was detected in the data stream.")
        print("Adaptive models are likely to perform better for this problem.")
    else:
        print("\nNo significant concept drift was detected in the data stream.")
        print("Simpler, non-adaptive models may be sufficient for this problem.")


if __name__ == "__main__":
    main()
