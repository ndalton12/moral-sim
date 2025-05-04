import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
    }
    data.append(row)

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

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

# Visualize model performance across ethical frameworks
plt.figure(figsize=(15, 10))

# Prepare data for plotting
ethical_metrics = ["utilitarian_mean", "maximin_mean", "virtue_mean"]
stderr_metrics = ["utilitarian_stderr", "maximin_stderr", "virtue_stderr"]
scenarios = df["scenario"].unique()

for i, scenario in enumerate(scenarios):
    scenario_df = df[df["scenario"] == scenario]

    plt.subplot(len(scenarios), 1, i + 1)
    plt.title(f"Scenario: {scenario}")

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
        ax = plot_data.plot(kind="bar", ax=plt.gca(), yerr=error_data, capsize=4)
        plt.ylabel("Score")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# Analyze outcomes for each scenario and model
def plot_outcomes(scenario_name):
    scenario_df = df[df["scenario"] == scenario_name]

    plt.figure(figsize=(15, 10))

    # Get all unique outcomes for this scenario
    all_outcomes = set()
    for outcomes_dict in scenario_df["outcomes"]:
        all_outcomes.update(outcomes_dict.keys())

    # Create a DataFrame with outcome counts for each model
    outcome_data = []
    for _, row in scenario_df.iterrows():
        model = row["model"]
        outcomes = row["outcomes"]
        n = row["utilitarian_n"]  # Using n as a total count reference

        for outcome in all_outcomes:
            count = outcomes.get(outcome, 0)
            percentage = count / n * 100
            outcome_data.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "count": count,
                    "percentage": percentage,
                }
            )

    outcome_df = pd.DataFrame(outcome_data)

    # Plot as a heatmap
    if len(outcome_df) > 0:
        pivot = outcome_df.pivot(index="model", columns="outcome", values="percentage")
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
        plt.title(f"Outcome Distribution for Scenario: {scenario_name}")
        plt.ylabel("Model")
        plt.xlabel("Outcome")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No outcome data available for scenario: {scenario_name}")


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

ethics_df["ethics_framework"] = ethics_df["ethics_framework"].str.replace("_mean", "")

plt.figure(figsize=(15, 8))
sns.boxplot(data=ethics_df, x="model", y="score", hue="ethics_framework")
plt.title("Ethical Framework Scores by Model")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
