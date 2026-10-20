
import pandas as pd

from difflib import SequenceMatcher
from itertools import combinations


# Load experimental responses
df = pd.read_csv(
    "results/full_experiments.csv",
    keep_default_na=False
)

print("Total Responses:", len(df))

print("\nResponses per Model:")
print(df["model"].value_counts())

print("\nResponses per Category:")
print(df["category"].value_counts())

print("\nMissing/Empty Responses:",
      df["response"].astype(str).str.strip().eq("").sum())

print("\nDuplicate Experiments:", df.duplicated(
    ["model", "configuration", "category", "run"]
).sum())

print("\nExperiment Groups:")
print(df.groupby(
    ["model", "configuration", "category"]
)["run"].nunique().value_counts())



results = []

for (model, config, category), group in df.groupby(
    ["model", "configuration", "category"]
):

    responses = group["response"].tolist()

    # Count different responses
    unique = len(set(responses))

    # Compare all 3 response pairs
    similarities = [
        SequenceMatcher(None, a, b).ratio()
        for a, b in combinations(responses, 2)
    ]

    average_similarity = sum(similarities) / len(similarities)

    results.append({
        "model": model,
        "configuration": config,
        "category": category,
        "unique_responses": unique,
        "similarity": average_similarity
    })


# Convert results to a DataFrame
summary = pd.DataFrame(results)

# Compare models and configurations
print("\n=== PARAMETER COMPARISON ===")

print(
    summary.groupby(["model", "configuration"])[
        ["unique_responses", "similarity"]
    ].mean().round(3)
)

# Save results
summary.to_csv(
    "results/variation_results.csv",
    index=False
)

print("\nAnalysis saved successfully!")



print("\n=== RESPONSE QUALITY REVIEW ===")

samples = df[
    (df["configuration"].isin(["temp_0", "temp_10"]))
    & (df["category"].isin(["factual", "reasoning"]))
    & (df["run"] == 1)
]

for _, row in samples.iterrows():
    print("\n------------------------------")
    print("Model:", row["model"])
    print("Configuration:", row["configuration"])
    print("Category:", row["category"])
    print("Response:")
    print(row["response"])
