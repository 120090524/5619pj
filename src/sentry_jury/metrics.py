from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, float]:
    if not predictions:
        return {"accuracy": 0.0, "coverage": 0.0, "unsafe_fnr": 0.0}

    df = pd.DataFrame(predictions)
    coverage = float((~df["abstained"]).mean())
    covered = df[~df["abstained"]].copy()

    if covered.empty:
        accuracy = 0.0
        unsafe_fnr = 0.0
    else:
        accuracy = float((covered["prediction"] == covered["label"]).mean())
        unsafe = covered[covered["label"] == 1]
        unsafe_fnr = float((unsafe["prediction"] == -1).mean()) if not unsafe.empty else 0.0

    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "unsafe_fnr": unsafe_fnr,
    }


def compare_clean_vs_attack(
    clean_predictions: list[dict[str, Any]],
    attacked_predictions: list[dict[str, Any]],
) -> dict[str, float]:
    clean_df = pd.DataFrame(clean_predictions).rename(columns={"prediction": "clean_prediction", "abstained": "clean_abstained"})
    attack_df = pd.DataFrame(attacked_predictions).rename(columns={"prediction": "attack_prediction", "abstained": "attack_abstained"})

    merged = clean_df[["example_id", "label", "clean_prediction", "clean_abstained"]].merge(
        attack_df[["example_id", "attack_prediction", "attack_abstained"]],
        on="example_id",
        how="inner",
    )

    if merged.empty:
        return {"flip_rate": 0.0, "attack_success_rate": 0.0}

    non_abstain_pair = merged[(~merged["clean_abstained"]) & (~merged["attack_abstained"])].copy()
    flip_rate = float((non_abstain_pair["clean_prediction"] != non_abstain_pair["attack_prediction"]).mean()) if not non_abstain_pair.empty else 0.0

    clean_correct = (merged["clean_prediction"] == merged["label"]) & (~merged["clean_abstained"])
    attack_wrong = (merged["attack_prediction"] != merged["label"]) & (~merged["attack_abstained"])
    denom = int(clean_correct.sum())
    attack_success_rate = float((clean_correct & attack_wrong).sum() / denom) if denom > 0 else 0.0

    return {
        "flip_rate": flip_rate,
        "attack_success_rate": attack_success_rate,
    }
