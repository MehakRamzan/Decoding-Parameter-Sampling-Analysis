MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

REPEATS = 3

BASE = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
}

CONFIGS = [
    {"name": "baseline", **BASE},

    # Temperature experiments
    {"name": "temp_0", **BASE, "temperature": 0.0},
    {"name": "temp_05", **BASE, "temperature": 0.5},
    {"name": "temp_10", **BASE, "temperature": 1.0},

    # Top-K experiments
    {"name": "topk_10", **BASE, "top_k": 10},
    {"name": "topk_80", **BASE, "top_k": 80},

    # Top-P experiments
    {"name": "topp_05", **BASE, "top_p": 0.5},
    {"name": "topp_10", **BASE, "top_p": 1.0}
]
