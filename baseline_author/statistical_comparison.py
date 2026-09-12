import pandas as pd
from scipy.stats import wilcoxon


# --------------------------------------------------
# 1. Load repeated CV fold results
# --------------------------------------------------

df = pd.read_csv("random_search_repeated_cv_folds.csv")


# --------------------------------------------------
# 2. Separate the two models
# --------------------------------------------------

baseline = df[
    df["model"] == "Original Baseline RF"
].set_index("fold")

random_search = df[
    df["model"] == "RandomizedSearch RF"
].set_index("fold")


# --------------------------------------------------
# 3. Metrics to compare
# --------------------------------------------------

metrics = [
    "accuracy",
    "f1",
    "auroc",
]


# --------------------------------------------------
# 4. Run paired Wilcoxon signed-rank tests
# --------------------------------------------------

results = []

for metric in metrics:

    baseline_values = baseline[metric]
    random_values = random_search[metric]

    statistic, p_value = wilcoxon(
        random_values,
        baseline_values,
        alternative="two-sided",
    )

    baseline_mean = baseline_values.mean()
    random_mean = random_values.mean()

    mean_difference = random_mean - baseline_mean

    results.append({
        "metric": metric,
        "baseline_mean": baseline_mean,
        "random_search_mean": random_mean,
        "mean_difference": mean_difference,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    })


# --------------------------------------------------
# 5. Create results DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 6. Holm multiple-comparison correction
# --------------------------------------------------
#
# We have 3 statistical tests:
#   Accuracy
#   F1
#   AUROC
#
# Holm correction controls the family-wise
# error rate at alpha = 0.05.
#
# --------------------------------------------------

p_values = results_df["p_value"].tolist()

number_of_tests = len(p_values)

# Sort p-values from smallest to largest
sorted_indices = sorted(
    range(number_of_tests),
    key=lambda i: p_values[i]
)

adjusted_p_values = [0.0] * number_of_tests

# Calculate Holm adjusted p-values
for rank, index in enumerate(sorted_indices):

    multiplier = number_of_tests - rank

    adjusted_p_values[index] = min(
        p_values[index] * multiplier,
        1.0
    )

# Enforce monotonicity of Holm-adjusted p-values
for rank in range(1, number_of_tests):

    previous_index = sorted_indices[rank - 1]
    current_index = sorted_indices[rank]

    adjusted_p_values[current_index] = max(
        adjusted_p_values[current_index],
        adjusted_p_values[previous_index]
    )


# Add corrected p-values
results_df["holm_adjusted_p"] = adjusted_p_values


# --------------------------------------------------
# 7. Determine statistical significance
# --------------------------------------------------

results_df["significant_0.05"] = (
    results_df["holm_adjusted_p"] < 0.05
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n==============================================")
print("STATISTICAL COMPARISON")
print("RandomizedSearch RF vs Original Baseline RF")
print("Paired Wilcoxon Signed-Rank Test")
print("Holm Multiple-Comparison Correction")
print("==============================================\n")

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 9. Print interpretation
# --------------------------------------------------

print("\n==============================================")
print("INTERPRETATION")
print("==============================================\n")

for _, row in results_df.iterrows():

    metric = row["metric"]
    adjusted_p = row["holm_adjusted_p"]
    difference = row["mean_difference"]

    if adjusted_p < 0.05:

        print(
            f"{metric.upper()}: statistically significant "
            f"(Holm-adjusted p = {adjusted_p:.6f}, "
            f"mean difference = {difference:.6f})"
        )

    else:

        print(
            f"{metric.upper()}: not statistically significant "
            f"(Holm-adjusted p = {adjusted_p:.6f}, "
            f"mean difference = {difference:.6f})"
        )


# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

output_file = "statistical_comparison_holm_results.csv"

results_df.to_csv(
    output_file,
    index=False,
)

print("\nResults saved to:")
print(output_file)