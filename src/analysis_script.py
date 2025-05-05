import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import textwrap

# Set style for plots
plt.style.use("ggplot")
sns.set(font_scale=1.2)

# Load the results file
with open(os.path.join(os.path.dirname(__file__), "../results.json"), "r") as f:
    results = json.load(f)

print(f"Loaded {len(results)} result entries")

# Convert to DataFrame for easier analysis
# Extract key metrics into flat structure
data = []
for entry in results:
    row = {
        "scenario": entry["scenario"],
        "model": entry["model"],
        "timestamp": entry["timestamp"],
        "utilitarian_mean": entry["statistics"]["utilitarian"]["mean"],
        "utilitarian_stderr": entry["statistics"]["utilitarian"]["stderr"],
        "utilitarian_n": entry["statistics"]["utilitarian"]["n"],
        "maximin_mean": entry["statistics"]["maximin"]["mean"],
        "maximin_stderr": entry["statistics"]["maximin"]["stderr"],
        "maximin_n": entry["statistics"]["maximin"]["n"],
        "virtue_mean": entry["statistics"]["virtue"]["mean"],
        "virtue_stderr": entry["statistics"]["virtue"]["stderr"],
        "virtue_n": entry["statistics"]["virtue"]["n"],
        "outcomes": entry["outcome_counts"],
        "decision_counts": entry["decision_counts"],
    }
    data.append(row)

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Define a custom order for models
model_order = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219-thinking",
    "random",
]

# Define display names for better visualization
model_display_names = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "o4-mini": "o4-mini",
    "claude-3-7-sonnet-20250219": "Claude-3-7-Sonnet",
    "claude-3-7-sonnet-20250219-thinking": "Claude-3-7-Sonnet (thinking)",
    "random": "Random",
}

# Create a categorical type with an ordered category
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

# Display the first few rows
df.head()

# Count unique scenarios and models
print(f"Number of unique scenarios: {df['scenario'].nunique()}")
print(f"Scenarios: {df['scenario'].unique()}")
print(f"Number of unique models: {df['model'].nunique()}")
print(f"Models: {df['model'].unique()}")

# Compare performance across models and ethical frameworks
pivot_df = df.pivot_table(
    index="model",
    columns="scenario",
    values=["utilitarian_mean", "maximin_mean", "virtue_mean"],
    aggfunc="mean",
)

# Prepare data for plotting
ethical_metrics = ["utilitarian_mean", "maximin_mean", "virtue_mean"]
stderr_metrics = ["utilitarian_stderr", "maximin_stderr", "virtue_stderr"]
scenarios = df["scenario"].unique()

# Visualize model performance across ethical frameworks - separate figures for each scenario
for scenario in scenarios:
    scenario_df = df[df["scenario"] == scenario].sort_values("model")
    plot_data = pd.DataFrame()
    error_data = pd.DataFrame()

    for j, metric in enumerate(ethical_metrics):
        metric_name = metric.split("_")[0].capitalize()
        stderr_metric = stderr_metrics[j]

        for model in scenario_df["model"].unique():
            model_data = scenario_df[scenario_df["model"] == model]
            if not model_data.empty:
                plot_data.loc[model, metric_name] = model_data[metric].values[0]
                error_data.loc[model, metric_name] = model_data[stderr_metric].values[0]

    if not plot_data.empty:
        ax = plot_data.plot(kind="bar", yerr=error_data, capsize=4)
        plt.title(f"Scenario: {scenario}", fontsize=16)
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("Model", fontsize=14)

        # Replace x-tick labels with display names
        ax.set_xticklabels(
            [model_display_names.get(m, m) for m in plot_data.index],
            rotation=45,
            ha="right",
        )

        # Improve legend visibility
        plt.legend(
            title="Ethical Framework",
            fontsize=12,
            title_fontsize=14,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


# Analyze outcomes for each scenario and model
def plot_outcomes(scenario_name):
    scenario_df = df[df["scenario"] == scenario_name].sort_values("model")

    plt.figure(figsize=(15, 10))

    # Get all unique outcomes for this scenario
    all_outcomes = set()
    for outcomes_dict in scenario_df["outcomes"]:
        all_outcomes.update(outcomes_dict.keys())

    # Create a DataFrame with outcome counts for each model
    outcome_data = []
    for _, row in scenario_df.iterrows():
        model = row["model"]
        if model == "random":
            continue
        outcomes = row["outcomes"]
        n = row["utilitarian_n"]  # Using n as a total count reference

        for outcome in all_outcomes:
            count = outcomes.get(outcome, 0)
            percentage = count / n * 100
            # Only include non-zero percentages
            if percentage > 0:
                outcome_data.append(
                    {
                        "model": model,
                        "model_display": model_display_names.get(model, model),
                        "outcome": outcome,
                        "count": count,
                        "percentage": percentage,
                    }
                )

    outcome_df = pd.DataFrame(outcome_data)

    # Plot as a heatmap
    if len(outcome_df) > 0:
        # Use model_display for the index
        pivot = outcome_df.pivot(
            index="model_display", columns="outcome", values="percentage"
        )
        # Sort according to the original model order
        model_display_order = [
            model_display_names.get(m, m) for m in model_order if m != "random"
        ]
        pivot = pivot.reindex(model_display_order)

        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
        plt.title(f"Outcome Distribution for Scenario: {scenario_name}", fontsize=16)
        plt.ylabel("Model", fontsize=14)
        plt.xlabel("Outcome", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No outcome data available for scenario: {scenario_name}")


# Analyze decision distributions for each scenario and model
def plot_decision_distributions(scenario_name):
    scenario_df = df[df["scenario"] == scenario_name].sort_values("model")

    # Skip if no data for this scenario
    if scenario_df.empty:
        print(f"No data available for scenario: {scenario_name}")
        return

    # Extract all decision points (node_ids) for this scenario
    all_decision_points = set()
    for _, row in scenario_df.iterrows():
        if "decision_counts" in row and row["decision_counts"]:
            all_decision_points.update(row["decision_counts"].keys())

    # Create a figure for each decision point
    for decision_point in sorted(all_decision_points):
        # Get all possible choices for this decision point
        all_choices = set()
        for _, row in scenario_df.iterrows():
            if (
                "decision_counts" in row
                and row["decision_counts"]
                and decision_point in row["decision_counts"]
            ):
                all_choices.update(row["decision_counts"][decision_point].keys())

        # Skip empty choice sets
        if not all_choices:
            continue

        # Collect data for each model
        models = []
        model_display_list = []
        choice_data = []

        for _, row in scenario_df.iterrows():
            model = row["model"]
            # Skip the random model for decision distributions
            if model == "random":
                continue

            if (
                "decision_counts" in row
                and row["decision_counts"]
                and decision_point in row["decision_counts"]
            ):
                models.append(model)
                model_display_list.append(model_display_names.get(model, model))

                # Get the choices and counts for this model
                choices_dict = row["decision_counts"][decision_point]
                row_data = []

                # For each possible choice, get the count or 0 if not present
                for choice in sorted(all_choices):
                    count = choices_dict.get(choice, 0)
                    # Convert to percentage
                    percentage = 100 * count / row["utilitarian_n"]
                    row_data.append(percentage)

                choice_data.append(row_data)

        # Skip if no models have data for this decision point
        if not models:
            continue

        # Convert to numpy array for easier manipulation
        choice_data = np.array(choice_data)

        # Sort models according to the predefined order
        sorted_indices = [
            i
            for i, _ in sorted(
                enumerate(models),
                key=lambda x: (
                    model_order.index(x[1]) if x[1] in model_order else len(model_order)
                ),
            )
        ]
        models = [models[i] for i in sorted_indices]
        model_display_list = [model_display_list[i] for i in sorted_indices]
        choice_data = choice_data[sorted_indices]

        # Setup the plot with appropriate size based on number of models and choices
        plt.figure(figsize=(14, max(6, len(models) * 0.7)))
        plt.title(f"Decision Distribution: {decision_point}", fontsize=16)
        plt.suptitle(f"Scenario: {scenario_name}", fontsize=14)

        # Set up the plot
        ax = plt.gca()
        y_pos = np.arange(len(models))

        # Plot stacked bars for each choice
        left = np.zeros(len(models))
        for i, choice in enumerate(sorted(all_choices)):
            # Choose a distinct color for each choice
            color = plt.cm.tab20(i / len(all_choices))
            ax.barh(
                y_pos,
                choice_data[:, i],
                left=left,
                height=0.7,
                label=textwrap.fill(choice, 40),
                color=color,
                alpha=0.8,
            )

            # Add percentage labels on the bars if they're significant enough
            for j, value in enumerate(choice_data[:, i]):
                if value > 5:  # Only label if percentage > 5%
                    text_x = left[j] + value / 2
                    ax.text(
                        text_x,
                        y_pos[j],
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                        fontweight="bold",
                    )

            left += choice_data[:, i]

        # Set y-axis ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_display_list, fontsize=12)

        # Set x-axis label
        ax.set_xlabel("Percentage of runs (%)", fontsize=14)

        # Add a legend outside the plot
        # Place the legend to the right of the plot
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=9,
            title="Choices",
            title_fontsize=11,
        )

        # Add grid lines for better readability
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Set x-axis limits
        plt.xlim(0, 100)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


# Plot outcomes for each scenario
for scenario in df["scenario"].unique():
    plot_outcomes(scenario)

# Compare ethical framework scores across models
ethics_df = df.melt(
    id_vars=["scenario", "model", "timestamp"],
    value_vars=["utilitarian_mean", "maximin_mean", "virtue_mean"],
    var_name="ethics_framework",
    value_name="score",
)

# Add display names
ethics_df["model_display"] = ethics_df["model"].apply(
    lambda x: model_display_names.get(x, x)
)
ethics_df["ethics_framework"] = ethics_df["ethics_framework"].str.replace("_mean", "")

plt.figure(figsize=(15, 8))
# Sort by the ordered model categorical
order_models = sorted(
    ethics_df["model"].unique(),
    key=lambda x: model_order.index(x) if x in model_order else len(model_order),
)
display_order = [model_display_names.get(m, m) for m in order_models]

sns.boxplot(
    data=ethics_df,
    x="model_display",
    y="score",
    hue="ethics_framework",
    order=display_order,
)
plt.title("Ethical Framework Scores by Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# Improve legend visibility
plt.legend(
    title="Ethical Framework",
    fontsize=12,
    title_fontsize=14,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.0,
)

plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot decision distributions for each scenario
print("\nAnalyzing decision distributions...")
for scenario in df["scenario"].unique():
    print(f"\nDecision distributions for scenario: {scenario}")
    plot_decision_distributions(scenario)
