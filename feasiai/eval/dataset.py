"""
Dataset loader: reads eval/dataset.jsonl and returns EvalSample objects.
"""

from __future__ import annotations

import json
from pathlib import Path

from feasiai.eval.schemas import EvalSample
from feasiai.schemas import FeasibilityLabel

_DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "dataset.jsonl"


def load_dataset(path: str | Path | None = None) -> list[EvalSample]:
    """
    Load the labeled evaluation dataset from a JSONL file.

    Args:
        path: Path to the JSONL file. Defaults to eval/dataset.jsonl.

    Returns:
        List of EvalSample objects.
    """
    dataset_path = Path(path) if path else _DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at: {dataset_path}. "
            "Run from the project root or specify the path explicitly."
        )

    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                # Parse label string to enum
                label_str = data.get("expected_label", "Unsupported / Unclear")
                label = _parse_label(label_str)
                samples.append(
                    EvalSample(
                        id=data.get("id", f"sample-{line_num}"),
                        requirement_text=data["requirement_text"],
                        expected_label=label,
                        section=data.get("section"),
                        is_ambiguous=data.get("is_ambiguous", False),
                        notes=data.get("notes", ""),
                        source=data.get("source", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Dataset parse error at line {line_num}: {e}") from e

    return samples


def _parse_label(label_str: str) -> FeasibilityLabel:
    mapping = {
        "Available": FeasibilityLabel.AVAILABLE,
        "Configurable": FeasibilityLabel.CONFIGURABLE,
        "Requires Development": FeasibilityLabel.REQUIRES_DEVELOPMENT,
        "Unsupported / Unclear": FeasibilityLabel.UNSUPPORTED_UNCLEAR,
    }
    if label_str not in mapping:
        raise ValueError(
            f"Unknown label '{label_str}'. Valid: {list(mapping.keys())}"
        )
    return mapping[label_str]

