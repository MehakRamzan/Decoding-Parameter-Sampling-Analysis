
import ollama
import pandas as pd
from pathlib import Path

from config import MODELS, CONFIGS, REPEATS
from prompts import PROMPTS


# ==========================================
# 1. EXPERIMENT SETTINGS
# ==========================================

TEST_MODE = False

models_to_run = MODELS[:1] if TEST_MODE else MODELS
configs_to_run = CONFIGS[:2] if TEST_MODE else CONFIGS

prompts_to_run = (
    list(PROMPTS.items())[:1]
    if TEST_MODE else list(PROMPTS.items())
)

repeats_to_run = 1 if TEST_MODE else REPEATS


# ==========================================
# 2. RESULTS FILE AND CHECKPOINTS
# ==========================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / (
    "pilot_results.csv" if TEST_MODE
    else "full_experiments.csv"
)

# Load previous results if they exist
if OUTPUT_FILE.exists():

    df_existing = pd.read_csv(
        OUTPUT_FILE,
        keep_default_na=False
    )

    records = df_existing.to_dict("records")

    print(f"Loaded {len(records)} previous responses.")

else:
    records = []
    print("Starting a new experiment.")


# Identify already completed experiments
completed = {
    (
        row["model"],
        row["configuration"],
        row["category"],
        int(row["run"])
    )
    for row in records
}


# ==========================================
# 3. BENCHMARKING EXPERIMENT
# ==========================================

total_experiments = (
    len(models_to_run)
    * len(configs_to_run)
    * len(prompts_to_run)
    * repeats_to_run
)

print(f"\nTotal planned experiments: {total_experiments}")


# Loop through models
for model in models_to_run:

    print(f"\n========== Model: {model} ==========")

    # Loop through decoding configurations
    for config in configs_to_run:

        print(f"\nConfiguration: {config['name']}")

        # Loop through prompt categories
        for category, prompt in prompts_to_run:

            print(f"\nTesting category: {category}")

            # Repeat each experiment
            for run in range(1, repeats_to_run + 1):

                run_key = (
                    model,
                    config["name"],
                    category,
                    run
                )

                # Skip experiments already completed
                if run_key in completed:

                    print(f"Already completed: Run {run}")
                    continue

                print(
                    f"Generating response {run}/{repeats_to_run}...",
                    flush=True
                )

                # Send request to Ollama
                response = ollama.chat(

                    model=model,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    options={
                        "temperature": config["temperature"],
                        "top_k": config["top_k"],
                        "top_p": config["top_p"],
                        "num_predict": 200
                    }
                )

                # Extract generated response
                text = response.message.content or ""

                # Store experiment information
                records.append({
                    "model": model,
                    "category": category,
                    "configuration": config["name"],
                    "temperature": config["temperature"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                    "run": run,
                    "response": text
                })

                # Save results after every response
                df = pd.DataFrame(records)

                df.to_csv(
                    OUTPUT_FILE,
                    index=False,
                    encoding="utf-8-sig"
                )

                # Mark experiment as completed
                completed.add(run_key)

                print(
                    f"Saved successfully! "
                    f"Completed: {len(completed)}/{total_experiments}"
                )

            # ==================================
            # 4. RESPONSE VARIATION
            # ==================================

            group_responses = [
                row["response"]
                for row in records
                if row["model"] == model
                and row["configuration"] == config["name"]
                and row["category"] == category
            ]

            unique_responses = len(set(group_responses))

            print("Total Responses:", len(group_responses))
            print("Unique Responses:", unique_responses)

            if len(group_responses) > 1:
                if unique_responses == 1:
                    print("All responses are identical.")
                else:
                    print("Response variation detected.")


# ==========================================
# 5. FINAL RESULTS
# ==========================================

df = pd.DataFrame(records)

print("\n========== EXPERIMENT COMPLETED ==========")

print("Dataset saved:", OUTPUT_FILE)
print("Dataset shape:", df.shape)

if not df.empty:
    print("\nResponses per model:")
    print(df.groupby("model").size())

    print("\nResponses per category:")
    print(df.groupby("category").size())

print("\nAll available results have been saved.")
