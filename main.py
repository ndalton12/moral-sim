# Main script to run simulations
import argparse
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from src.moral_sim.tree import DecisionTree
from src.moral_sim.simulator import SimulationRunner, SimulationError
from src.utils import (
    load_api_keys,
    calculate_statistics,
    calculate_outcome_distribution,
    append_results_to_json,
)


async def run_single_simulation(runner, run_num=None, num_runs=None):
    """
    Run a single simulation and return the outcome.

    Args:
        runner: SimulationRunner instance
        run_num: Current run number (optional, for display purposes)
        num_runs: Total number of runs (optional, for display purposes)

    Returns:
        Outcome result from the simulation
    """
    try:
        # Show run info if multiple runs
        if run_num is not None and num_runs is not None and num_runs > 1:
            print(f"Run {run_num}/{num_runs}")

        # Run the simulation asynchronously
        outcome_result = await runner.run_async()

        return outcome_result

    except Exception as e:
        print(f"Error in run {run_num if run_num else ''}: {e}")
        return None


async def run_multiple_simulations(
    tree, model, num_runs, include_history=True, max_history_items=5
):
    """
    Run multiple simulations in parallel using asyncio.

    Args:
        tree: DecisionTree instance
        model: Name of the LLM model to use
        num_runs: Number of simulations to run
        include_history: Whether to include conversation history
        max_history_items: Maximum number of history items to include

    Returns:
        List of outcome results from all simulations
    """
    # Create a runner for each simulation
    runners = [
        SimulationRunner(
            tree,
            model=model,
            include_history=include_history,
            max_history_items=max_history_items,
        )
        for _ in range(num_runs)
    ]

    # Run all simulations in parallel with progress bar
    tasks = [
        run_single_simulation(runner, i + 1, num_runs)
        for i, runner in enumerate(runners)
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Running simulations")

    return results


def process_and_save_results(all_outcomes, scenario_name, model_name, output_file):
    """
    Process simulation results and save them to a file.

    Args:
        all_outcomes: List of outcome results from simulations
        scenario_name: Name of the scenario
        model_name: Name of the model used
        output_file: Path to save results

    Returns:
        Tuple of (statistics, outcome_distribution)
    """
    # For single run, use simplified statistics
    if len(all_outcomes) == 1 and all_outcomes[0]:
        outcome = all_outcomes[0]

        # Create simplified statistics for single run
        statistics = {}
        if "scores" in outcome:
            for ideology, score in outcome["scores"].items():
                statistics[ideology] = {"mean": float(score), "stderr": 0.0, "n": 1}

        # Create outcome distribution for single run
        outcome_distribution = {}
        if "node_id" in outcome:
            outcome_distribution[outcome["node_id"]] = 1

    # For multiple runs, calculate statistics
    else:
        statistics = calculate_statistics(all_outcomes)
        outcome_distribution = calculate_outcome_distribution(all_outcomes)

        # Print the results
        print(f"Completed {len(all_outcomes)} simulation runs")

        # Print ideology statistics
        for ideology, stats in statistics.items():
            print(
                f"{ideology}: mean={stats['mean']:.2f}, stderr={stats['stderr']:.2f}, n={stats['n']}"
            )

        # Print outcome distribution
        print("Outcome distribution:")
        for node_id, count in outcome_distribution.items():
            print(f"  {node_id}: {count} runs")

    # Save results
    append_results_to_json(
        statistics, outcome_distribution, scenario_name, model_name, output_file
    )
    print(f"Results appended to {output_file}")

    return statistics, outcome_distribution


async def main_async():
    parser = argparse.ArgumentParser(description="Run an LLM-driven moral simulation.")
    parser.add_argument("scenario_file", help="Path to the scenario YAML file.")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Name of the LLM model to use (via LiteLLM).",
    )
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to the API keys JSON file (default: api_keys.json)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable including conversation history in LLM context",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=5,
        help="Maximum number of history items to include in LLM context (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Number of simulation runs to perform (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results.json",
        help="Path to save results (default: results.json)",
    )

    args = parser.parse_args()

    try:
        load_api_keys(args.api_key_file)

        if not os.path.exists(args.scenario_file):
            print(f"Error: Scenario file not found at {args.scenario_file}")
            return

        # Extract scenario name from file path for reporting
        scenario_name = os.path.splitext(os.path.basename(args.scenario_file))[0]

        print(f"Loading scenario from {args.scenario_file}")
        tree = DecisionTree.load_from_yaml(args.scenario_file)

        print(f"Initializing simulation with model {args.model}")

        # Run simulations in parallel
        print(f"Starting {args.num_runs} simulation run(s)")
        all_outcomes = await run_multiple_simulations(
            tree,
            args.model,
            args.num_runs,
            include_history=not args.no_history,
            max_history_items=args.max_history,
        )

        # Process and save results
        process_and_save_results(all_outcomes, scenario_name, args.model, args.output)

    except SimulationError as e:
        print(f"Error: Simulation Error: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find scenario file '{args.scenario_file}'")
    except Exception as e:
        import traceback

        print(f"Error: An unexpected error occurred: {e}")
        traceback.print_exc()


def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
