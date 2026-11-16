
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "3D CNN V2",
        "Spatial ViT",
        "Hybrid CNN-Transformer",
    ]

    metrics = {
        "Accuracy": [0.9890, 0.9932, 1.0000],
        "Precision": [0.9417, 0.9597, 1.0000],
        "Recall": [0.9826, 0.9931, 1.0000],
        "F1-score": [0.9617, 0.9761, 1.0000],
    }

    protocols = [
        "Random split\n(820 test samples)",
        "Spatial holdout\n(1,024 test samples)",
        "Spatial holdout\n(1,024 test samples)",
    ]

    metric_names = list(metrics.keys())
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 7))

    for index, metric in enumerate(metric_names):
        values = metrics[metric]
        positions = x + (index - 1.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=metric,
        )

        ax.bar_label(
            bars,
            labels=[f"{value * 100:.1f}%" for value in values],
            padding=3,
            fontsize=8,
        )

    ax.set_title(
        "TerraSpectra: Model Performance Summary",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_xlabel("Model and evaluation protocol", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{model}\n{protocol}"
            for model, protocol in zip(models, protocols)
        ],
        fontsize=9,
    )

    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{value}%" for value in range(0, 101, 20)])

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=4, frameon=False)

    fig.text(
        0.5,
        0.015,
        (
            "IMPORTANT: Results use synthetic mock data. "
            "Evaluation splits and test-set sizes differ; "
            "this is a descriptive summary, not a controlled benchmark "
            "or evidence of real-world crop disease detection."
        ),
        ha="center",
        fontsize=9,
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])

    output_path = output_dir / "model_performance_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Model comparison visualization generated successfully.")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
