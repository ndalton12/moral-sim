import json
import os
from collections import defaultdict, Counter
import numpy as np


def load_api_keys_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def load_api_keys(file_path):
    api_keys = load_api_keys_from_file(file_path)
    for key, value in api_keys.items():
        os.environ[key] = value


def calculate_statistics(outcomes_list):
    """
    Calculate mean and standard error for each ideology score across multiple runs.

    Args:
        outcomes_list: List of outcome dictionaries from multiple simulation runs

    Returns:
        Dictionary with mean and standard error for each ideology
    """
    # Create a dictionary to store all scores for each ideology
    ideology_scores = defaultdict(list)

    # Collect scores for each ideology across all runs
    for outcome in outcomes_list:
        if outcome and "scores" in outcome:  # Skip runs that didn't reach an outcome
            for ideology, score in outcome["scores"].items():
                ideology_scores[ideology].append(score)

    # Calculate statistics for each ideology
    statistics = {}
    for ideology, scores in ideology_scores.items():
        if scores:  # Only calculate if we have scores
            scores_array = np.array(scores)
            mean = np.mean(scores_array)
            # Use standard error of the mean if we have enough samples
            stderr = (
                np.std(scores_array, ddof=1) / np.sqrt(len(scores_array))
                if len(scores_array) > 1
                else 0
            )

            statistics[ideology] = {
                "mean": float(mean),
                "stderr": float(stderr),
                "n": len(scores_array),
            }

    return statistics


def calculate_outcome_distribution(outcomes_list):
    """
    Calculate the distribution of outcome nodes reached across simulation runs.

    Args:
        outcomes_list: List of outcome dictionaries from multiple simulation runs

    Returns:
        Dictionary with count for each outcome node
    """
    # Count occurrences of each outcome node
    outcome_counts = Counter()

    for outcome in outcomes_list:
        if outcome and "node_id" in outcome:
            outcome_counts[outcome["node_id"]] += 1

    # Return counts as a simple dictionary
    return dict(outcome_counts)


def calculate_decision_distribution(outcomes_list):
    """
    Calculate the distribution of decisions made across simulation runs.

    Args:
        outcomes_list: List of outcome dictionaries from multiple simulation runs

    Returns:
        Dictionary with counts for each decision (node_id -> {choice_text -> count})
    """
    # Create a nested dictionary for node_id -> choice_text -> count
    decision_counts = defaultdict(Counter)

    for outcome in outcomes_list:
        if outcome and "decisions" in outcome:
            for node_id, choice_text in outcome["decisions"].items():
                decision_counts[node_id][choice_text] += 1

    # Convert to regular dict for JSON serialization
    return {
        node_id: dict(choice_counts)
        for node_id, choice_counts in decision_counts.items()
    }


def append_results_to_json(
    statistics,
    outcome_distribution,
    scenario_name,
    model_name,
    output_file="results.json",
    decision_distribution=None,
):
    """
    Append simulation statistics to a JSON file. If the file doesn't exist, it will be created.

    Args:
        statistics: Dictionary with statistics for each ideology
        outcome_distribution: Dictionary with counts of outcome nodes
        scenario_name: Name of the scenario
        model_name: Name of the model used
        output_file: Path to the output JSON file (default: results.json)
        decision_distribution: Dictionary with counts of decisions made at each node
    """
    # Create a result entry for this run
    new_result = {
        "scenario": scenario_name,
        "model": model_name,
        "timestamp": import_datetime_if_needed(),
        "statistics": statistics,
        "outcome_counts": outcome_distribution,
    }

    # Add decision distribution if available
    if decision_distribution:
        new_result["decision_counts"] = decision_distribution

    # Load existing results or create a new results list
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "r") as f:
                results = json.load(f)
                if not isinstance(results, list):
                    results = [results]  # Convert to list if it's a single result
        except json.JSONDecodeError:
            # If the file is corrupted, start with a new list
            results = []
    else:
        results = []

    # Append the new result
    results.append(new_result)

    # Save the updated results back to the file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def import_datetime_if_needed():
    """Import datetime and return current timestamp in ISO format."""
    from datetime import datetime

    return datetime.now().isoformat()
