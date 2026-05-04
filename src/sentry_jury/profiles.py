from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .utils import clamp


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        agreement = float(np.mean(a == b))
        return max(0.0, 2.0 * agreement - 1.0)
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        agreement = float(np.mean(a == b))
        return max(0.0, 2.0 * agreement - 1.0)
    return float(max(0.0, abs(corr)))


def compute_dependence(clean_df: pd.DataFrame) -> dict[str, float]:
    pivot = clean_df.pivot(index="example_id", columns="sensor", values="prediction")
    sensors = list(pivot.columns)
    dependence: dict[str, float] = {}
    for sensor in sensors:
        values = []
        for other in sensors:
            if other == sensor:
                continue
            joined = pivot[[sensor, other]].dropna()
            if joined.empty:
                continue
            corr = _safe_correlation(joined[sensor].to_numpy(), joined[other].to_numpy())
            values.append(corr)
        dependence[sensor] = float(np.mean(values)) if values else 0.0
    return dependence


def compute_profiles(
    clean_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    attack_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    sensors = sorted(clean_df["sensor"].unique().tolist())
    dependence = compute_dependence(clean_df)

    profiles: dict[str, dict[str, Any]] = {}
    invariance_df = probe_df[probe_df["kind"] == "invariance"].copy()

    for sensor in sensors:
        clean_sensor = clean_df[clean_df["sensor"] == sensor].copy()
        accuracy = float((clean_sensor["prediction"] == clean_sensor["label"]).mean())

        if invariance_df.empty:
            consistency = 1.0
        else:
            merged = clean_sensor[["example_id", "prediction"]].rename(columns={"prediction": "clean_prediction"}).merge(
                invariance_df[invariance_df["sensor"] == sensor][["example_id", "prediction"]],
                on="example_id",
                how="inner",
            )
            consistency = float((merged["clean_prediction"] == merged["prediction"]).mean()) if not merged.empty else 1.0

        r = clamp(0.55 * accuracy + 0.45 * consistency, 0.501, 0.999)
        base_weight = float(math.log(r / (1.0 - r)))

        vulnerabilities: dict[str, float] = {}
        attack_names = sorted(attack_df["attack_name"].unique().tolist()) if not attack_df.empty else []
        clean_error = float((clean_sensor["prediction"] != clean_sensor["label"]).mean())

        for attack_name in attack_names:
            attacked_sensor = attack_df[(attack_df["sensor"] == sensor) & (attack_df["attack_name"] == attack_name)].copy()
            merged = clean_sensor[["example_id", "label", "prediction"]].rename(columns={"prediction": "clean_prediction"}).merge(
                attacked_sensor[["example_id", "prediction"]].rename(columns={"prediction": "attack_prediction"}),
                on="example_id",
                how="inner",
            )
            if merged.empty:
                vulnerabilities[attack_name] = 0.0
                continue
            attack_error = float((merged["attack_prediction"] != merged["label"]).mean())
            flip_rate = float((merged["attack_prediction"] != merged["clean_prediction"]).mean())
            # On very small calibration sets, pure error-increase estimates are noisy.
            # We therefore blend attack-induced error increase with a smaller flip-rate term.
            value = max(0.0, attack_error - clean_error) + 0.25 * flip_rate
            vulnerabilities[attack_name] = float(min(1.0, value))

        profiles[sensor] = {
            "accuracy": accuracy,
            "consistency": consistency,
            "base_reliability": r,
            "base_weight": base_weight,
            "dependence": dependence.get(sensor, 0.0),
            "vulnerabilities": vulnerabilities,
        }

    return profiles
