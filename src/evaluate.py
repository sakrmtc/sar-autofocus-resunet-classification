from __future__ import annotations

import json
from pathlib import Path


def read_json(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Missing file: {path}")
        return None
    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    autofocus_metrics = read_json("./results/metrics/autofocus_metrics.json")
    classifier_metrics = read_json("./results/metrics/classifier_metrics.json")

    print("\n=== Evaluation Summary ===")
    if autofocus_metrics:
        print("Autofocus metrics:")
        for k, v in autofocus_metrics.items():
            print(f"  - {k}: {v}")

    if classifier_metrics:
        print("\nClassifier metrics:")
        for k, v in classifier_metrics.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
