import pandas as pd
import matplotlib.pyplot as plt

# Load our analysis results
df = pd.read_csv("results/variation_results.csv")

# Temperature configurations
temperatures = {
    "temp_0": 0.0,
    "temp_05": 0.5,
    "baseline": 0.7,
    "temp_10": 1.0
}

# Select temperature experiments
temp_df = df[df["configuration"].isin(temperatures)].copy()

temp_df["temperature"] = temp_df["configuration"].map(temperatures)

# Calculate average similarity for each model
summary = temp_df.groupby(
    ["model", "temperature"]
)["similarity"].mean().reset_index()

# Create graph
for model, group in summary.groupby("model"):

    group = group.sort_values("temperature")

    plt.plot(
        group["temperature"],
        group["similarity"],
        marker="o",
        label=model
    )

plt.title("Effect of Temperature on Response Similarity")
plt.xlabel("Temperature")
plt.ylabel("Average Text Similarity")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

# Save graph
plt.savefig(
    "results/temperature_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ====================================
# TOP-K COMPARISON
# ====================================

topk_values = {
    "topk_10": 10,
    "baseline": 40,
    "topk_80": 80
}

topk_df = df[
    df["configuration"].isin(topk_values)
].copy()

topk_df["top_k"] = topk_df["configuration"].map(topk_values)

summary = topk_df.groupby(
    ["model", "top_k"]
)["similarity"].mean().reset_index()

# Create a separate graph
plt.figure(figsize=(8, 5))

for model, group in summary.groupby("model"):

    group = group.sort_values("top_k")

    plt.plot(
        group["top_k"],
        group["similarity"],
        marker="o",
        label=model
    )

plt.title("Effect of Top-K on Response Similarity")
plt.xlabel("Top-K")
plt.ylabel("Average Text Similarity")
plt.xticks([10, 40, 80])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(
    "results/topk_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ====================================
# TOP-P COMPARISON
# ====================================

topp_values = {
    "topp_05": 0.5,
    "baseline": 0.9,
    "topp_10": 1.0
}

topp_df = df[
    df["configuration"].isin(topp_values)
].copy()

topp_df["top_p"] = topp_df["configuration"].map(topp_values)

summary = topp_df.groupby(
    ["model", "top_p"]
)["similarity"].mean().reset_index()

# Create graph
plt.figure(figsize=(8, 5))

for model, group in summary.groupby("model"):

    group = group.sort_values("top_p")

    plt.plot(
        group["top_p"],
        group["similarity"],
        marker="o",
        label=model
    )

plt.title("Effect of Top-P on Response Similarity")
plt.xlabel("Top-P")
plt.ylabel("Average Text Similarity")
plt.xticks([0.5, 0.9, 1.0])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(
    "results/topp_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
