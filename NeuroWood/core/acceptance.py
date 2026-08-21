from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_ACCEPTANCE = {
    "small_max_area_pct": 1.0,
    "medium_max_area_pct": 3.0,
    "max_small": 3,
    "max_medium": 2,
    "max_large": 1,
}


def load_acceptance_config(
    config: dict[str, Any] | None,
    fallback_path: str | Path = "config_acceptance.yaml",
) -> dict[str, Any]:
    """Lê a regra de aceitação do config principal ou de um YAML dedicado."""
    acceptance = None

    if isinstance(config, dict):
        candidate = config.get("acceptance")
        if isinstance(candidate, dict):
            acceptance = candidate

    if acceptance is None:
        path = Path(fallback_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                acceptance = loaded.get("acceptance", loaded)

    merged = dict(DEFAULT_ACCEPTANCE)
    if isinstance(acceptance, dict):
        merged.update(acceptance)

    _validate_acceptance_config(merged)
    return merged


def evaluate_acceptance(
    summary: dict[str, Any],
    acceptance: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Classifica os defeitos por área percentual do patch e aplica os limites.

    Pequeno: area_pct <= small_max_area_pct
    Médio : small_max_area_pct < area_pct <= medium_max_area_pct
    Grande: area_pct > medium_max_area_pct
    """
    small_max = float(acceptance["small_max_area_pct"])
    medium_max = float(acceptance["medium_max_area_pct"])

    counts = {"small": 0, "medium": 0, "large": 0}

    for patch in summary.get("patches", []):
        for defect in patch.get("defects", []):
            area_pct = float(defect.get("area_pct", 0.0) or 0.0)

            if area_pct <= small_max:
                size_class = "small"
            elif area_pct <= medium_max:
                size_class = "medium"
            else:
                size_class = "large"

            defect["size_class"] = size_class
            counts[size_class] += 1

    limits = {
        "small": int(acceptance["max_small"]),
        "medium": int(acceptance["max_medium"]),
        "large": int(acceptance["max_large"]),
    }

    exceeded = {
        size: counts[size] > limits[size]
        for size in ("small", "medium", "large")
    }

    accepted = not any(exceeded.values())
    classification = "ACCEPT" if accepted else "REJECT"

    details = {
        "accepted": accepted,
        "classification": classification,
        "counts": counts,
        "limits": limits,
        "thresholds": {
            "small_max_area_pct": small_max,
            "medium_max_area_pct": medium_max,
        },
        "exceeded": exceeded,
    }

    summary["acceptance"] = details
    return classification, details


def _validate_acceptance_config(cfg: dict[str, Any]) -> None:
    small_max = float(cfg["small_max_area_pct"])
    medium_max = float(cfg["medium_max_area_pct"])

    if small_max < 0:
        raise ValueError("acceptance.small_max_area_pct deve ser >= 0.")
    if medium_max <= small_max:
        raise ValueError(
            "acceptance.medium_max_area_pct deve ser maior que small_max_area_pct."
        )

    for key in ("max_small", "max_medium", "max_large"):
        value = int(cfg[key])
        if value < 0:
            raise ValueError(f"acceptance.{key} deve ser >= 0.")
