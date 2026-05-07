from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .aggregator import (
    aggregate_clean_weighted,
    aggregate_majority,
    aggregate_selectively,
    aggregate_single_sensor,
)
from .attacks import Attack, build_attacks
from .datasets import load_examples, split_calibration_test
from .meta_aggregator import MetaAggregator
from .metrics import compare_clean_vs_attack, summarize_predictions
from .profiles import compute_profiles
from .probes import Probe, build_probes
from .types import EvalExample, Sensor
from .utils import ensure_dir, write_json, write_jsonl
from .judges import build_judge


class ExperimentRunner:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        self.output_dir = ensure_dir(config.get("output_dir", "runs/default"))

        dataset_cfg = config["dataset"]
        self.examples, self.calibration_examples, self.test_examples = self._load_dataset_splits(dataset_cfg)

        calibration_limit = dataset_cfg.get("calibration_limit")
        test_limit = dataset_cfg.get("test_limit")
        if calibration_limit is not None:
            self.calibration_examples = self.calibration_examples[: int(calibration_limit)]
        if test_limit is not None:
            self.test_examples = self.test_examples[: int(test_limit)]

        self.examples = self.calibration_examples + self.test_examples

        self.prompt_names: list[str] = list(config["prompts"])
        self.order_variants: list[str] = list(config.get("order_variants", ["original"]))
        self.judges = {spec["name"]: build_judge(spec) for spec in config["judges"]}

        attacks_target_field = str(config.get("attacks_target_field", "response_a"))
        self.attacks: list[Attack] = build_attacks(list(config.get("attacks", [])), target_field=attacks_target_field)
        self.probes: list[Probe] = build_probes(list(config.get("probes", [])))
        self.sensors: list[Sensor] = self._build_sensors()

        self.lambda_attack = float(config["aggregation"]["lambda_attack"])
        self.rho_dependence = float(config["aggregation"]["rho_dependence"])
        self.margin = float(config["aggregation"]["margin"])

        self.methods: list[str] = list(
            config.get("methods", ["single_best", "majority_vote", "clean_weighted", "sentry"])
        )

        self.enable_prediction_cache = bool(config.get("enable_prediction_cache", True))
        self._disk_cache_path = self.output_dir / "prediction_cache.json"
        self._prediction_cache: dict[tuple[str, str], int] = self._load_disk_cache()
        self._cache_hits = 0
        self._cache_misses = 0

        self._meta_aggregator: MetaAggregator | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentRunner":
        with Path(path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _load_dataset_splits(
        self,
        dataset_cfg: dict[str, Any],
    ) -> tuple[list[EvalExample], list[EvalExample], list[EvalExample]]:
        calibration_path = dataset_cfg.get("calibration_path")
        test_path = dataset_cfg.get("test_path")

        if calibration_path and test_path:
            calibration_examples = load_examples(calibration_path)
            test_examples = load_examples(test_path)
            all_examples = calibration_examples + test_examples
            return all_examples, calibration_examples, test_examples

        all_examples = load_examples(dataset_cfg["path"])
        calibration_examples, test_examples = split_calibration_test(
            all_examples,
            calibration_fraction=float(dataset_cfg.get("calibration_fraction", 0.5)),
            seed=self.seed,
        )
        return all_examples, calibration_examples, test_examples

    def _build_sensors(self) -> list[Sensor]:
        sensors: list[Sensor] = []
        for judge_name in self.judges:
            for prompt_name in self.prompt_names:
                for order_variant in self.order_variants:
                    sensors.append(
                        Sensor(
                            judge_name=judge_name,
                            prompt_name=prompt_name,
                            order_variant=order_variant,
                        )
                    )
        return sensors

    def _apply_order_variant(self, example: EvalExample, sensor: Sensor) -> tuple[EvalExample, int]:
        if sensor.order_variant == "original":
            return example, 1
        if sensor.order_variant == "swapped":
            if example.task_type != "pairwise":
                return example, 1
            if example.response_b is None:
                raise ValueError("pairwise example missing response_b")
            from dataclasses import replace
            return replace(example, response_a=example.response_b, response_b=example.response_a), -1
        raise ValueError(f"Unknown order variant: {sensor.order_variant}")

    def _load_disk_cache(self) -> dict[tuple[str, str], int]:
        if not self.enable_prediction_cache or not self._disk_cache_path.exists():
            return {}
        try:
            with self._disk_cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded = {tuple(k.split("|||", 1)): v for k, v in raw.items()}
            print(f"[cache] Loaded {len(loaded)} entries from disk cache.")
            return loaded
        except Exception:
            return {}

    def _save_disk_cache(self) -> None:
        if not self.enable_prediction_cache:
            return
        raw = {"|||".join(k): v for k, v in self._prediction_cache.items()}
        with self._disk_cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f)

    def _example_cache_key(self, example: EvalExample) -> str:
        payload = example.to_dict()
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _sensor_predict(self, example: EvalExample, sensor: Sensor) -> int:
        cache_key = (sensor.key, self._example_cache_key(example))
        if self.enable_prediction_cache and cache_key in self._prediction_cache:
            self._cache_hits += 1
            return self._prediction_cache[cache_key]

        self._cache_misses += 1
        judge = self.judges[sensor.judge_name]
        ordered_example, sign_adjustment = self._apply_order_variant(example, sensor)
        result = judge.predict(ordered_example, sensor.prompt_name)
        decision = int(result.decision)
        if decision not in {-1, 1}:
            raise ValueError(f"Judge returned invalid decision {decision} for sensor {sensor.key}")

        final_decision = sign_adjustment * decision
        if self.enable_prediction_cache:
            self._prediction_cache[cache_key] = final_decision
            self._save_disk_cache()
        return final_decision

    def _collect_clean_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="clean calibration", leave=False):
            for sensor in self.sensors:
                pred = self._sensor_predict(example, sensor)
                rows.append({
                    "example_id": example.example_id,
                    "sensor": sensor.key,
                    "prediction": pred,
                    "label": example.label,
                })
        return rows

    def _collect_probe_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="probe calibration", leave=False):
            for probe in self.probes:
                probe_result = probe.apply(example)
                for sensor in self.sensors:
                    pred = self._sensor_predict(probe_result.example, sensor)
                    rows.append({
                        "example_id": example.example_id,
                        "sensor": sensor.key,
                        "probe_name": probe.name,
                        "family": probe.family,
                        "kind": probe.kind,
                        "prediction": pred,
                        "label": example.label,
                    })
        return rows

    def _collect_attack_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="attack calibration", leave=False):
            for attack in self.attacks:
                attacked = attack.apply(example)
                for sensor in self.sensors:
                    pred = self._sensor_predict(attacked.example, sensor)
                    rows.append({
                        "example_id": example.example_id,
                        "sensor": sensor.key,
                        "attack_name": attack.name,
                        "family": attack.family,
                        "prediction": pred,
                        "label": example.label,
                    })
        return rows

    def _empty_clean_df(self):
        import pandas as pd
        return pd.DataFrame(columns=["example_id", "sensor", "prediction", "label"])

    def _empty_probe_df(self):
        import pandas as pd
        return pd.DataFrame(
            columns=["example_id", "sensor", "probe_name", "family", "kind", "prediction", "label"]
        )

    def _empty_attack_df(self):
        import pandas as pd
        return pd.DataFrame(columns=["example_id", "sensor", "attack_name", "family", "prediction", "label"])

    def calibrate(self) -> dict[str, dict[str, Any]]:
        import pandas as pd

        clean_rows = self._collect_clean_rows(self.calibration_examples)
        probe_rows = self._collect_probe_rows(self.calibration_examples)
        attack_rows = self._collect_attack_rows(self.calibration_examples)

        clean_df = pd.DataFrame(clean_rows) if clean_rows else self._empty_clean_df()
        probe_df = pd.DataFrame(probe_rows) if probe_rows else self._empty_probe_df()
        attack_df = pd.DataFrame(attack_rows) if attack_rows else self._empty_attack_df()

        profiles = compute_profiles(clean_df=clean_df, probe_df=probe_df, attack_df=attack_df)

        write_jsonl(self.output_dir / "calibration_clean_rows.jsonl", clean_rows)
        write_jsonl(self.output_dir / "calibration_probe_rows.jsonl", probe_rows)
        write_jsonl(self.output_dir / "calibration_attack_rows.jsonl", attack_rows)
        write_json(self.output_dir / "profiles.json", profiles)

        # Train meta-aggregator if meta_sentry is requested
        if "meta_sentry" in self.methods:
            probe_votes_by_example: dict[str, dict[str, dict[str, int]]] = {}
            for row in probe_rows:
                eid = row["example_id"]
                family = row.get("family", "unknown")
                sensor = row["sensor"]
                vote = int(row["prediction"])
                probe_votes_by_example.setdefault(eid, {}).setdefault(family, {})[sensor] = vote

            coverage_floor = float(self.config.get("meta_sentry_coverage_floor", 0.80))
            self._meta_aggregator = MetaAggregator(coverage_floor=coverage_floor)
            self._meta_aggregator.fit(
                calibration_rows=clean_rows,
                profiles=profiles,
                probe_votes_by_example=probe_votes_by_example,
            )

        return profiles

    def _collect_votes(self, example: EvalExample) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        base_votes: dict[str, int] = {}
        probe_votes_by_family: dict[str, dict[str, int]] = {}

        for sensor in self.sensors:
            base_votes[sensor.key] = self._sensor_predict(example, sensor)

        for probe in self.probes:
            probe_result = probe.apply(example)
            family_votes: dict[str, int] = {}
            for sensor in self.sensors:
                family_votes[sensor.key] = self._sensor_predict(probe_result.example, sensor)
            probe_votes_by_family[probe.family] = family_votes

        return base_votes, probe_votes_by_family

    def _predict_example_with_method(
        self,
        example: EvalExample,
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> dict[str, Any]:
        base_votes, probe_votes_by_family = self._collect_votes(example)

        if method == "single_best":
            agg = aggregate_single_sensor(base_votes, best_sensor)
        elif method == "majority_vote":
            agg = aggregate_majority(base_votes)
        elif method == "clean_weighted":
            agg = aggregate_clean_weighted(base_votes, profiles, margin=self.margin)
        elif method == "sentry":
            agg = aggregate_selectively(
                base_votes=base_votes,
                profiles=profiles,
                probe_votes_by_family=probe_votes_by_family,
                lambda_attack=self.lambda_attack,
                rho_dependence=self.rho_dependence,
                margin=self.margin,
            )
        elif method == "meta_sentry":
            if self._meta_aggregator is None:
                raise RuntimeError("meta_sentry requires calibration first.")
            agg = self._meta_aggregator.aggregate(
                base_votes=base_votes,
                profiles=profiles,
                probe_votes_by_family=probe_votes_by_family,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "example_id": example.example_id,
            "label": example.label,
            "prediction": agg.prediction,
            "abstained": agg.abstained,
            "score": agg.score,
            "confidence": agg.confidence,
            "instance_risks": agg.instance_risks,
            "sensor_weights": agg.sensor_weights,
        }

    def _evaluate_split(
        self,
        examples: list[EvalExample],
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc=f"{method}", leave=False):
            rows.append(self._predict_example_with_method(example, profiles, method, best_sensor))
        return rows

    def _evaluate_attacks(
        self,
        examples: list[EvalExample],
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> dict[str, list[dict[str, Any]]]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        for attack in self.attacks:
            rows: list[dict[str, Any]] = []
            for example in tqdm(examples, desc=f"{method}:{attack.name}", leave=False):
                attacked = attack.apply(example)
                pred = self._predict_example_with_method(attacked.example, profiles, method, best_sensor)
                pred["attack_name"] = attack.name
                rows.append(pred)
            outputs[attack.name] = rows
        return outputs

    def run(self) -> dict[str, Any]:
        profiles = self.calibrate()
        if not profiles:
            raise ValueError("No sensor profiles were computed. Check judges / prompts / dataset.")

        best_sensor = max(profiles.items(), key=lambda kv: kv[1]["base_weight"])[0]

        summary = {
            "project_name": self.config.get("project_name", "sentry_jury"),
            "num_examples_total": len(self.examples),
            "num_examples_calibration": len(self.calibration_examples),
            "num_examples_test": len(self.test_examples),
            "num_sensors": len(self.sensors),
            "best_sensor": best_sensor,
            "methods": {},
            "cache": {
                "enabled": self.enable_prediction_cache,
                "hits": 0,
                "misses": 0,
                "size": 0,
            },
        }

        for method in self.methods:
            clean_predictions = self._evaluate_split(self.test_examples, profiles, method, best_sensor)
            write_jsonl(self.output_dir / f"{method}_test_clean_predictions.jsonl", clean_predictions)
            clean_metrics = summarize_predictions(clean_predictions)

            attacked_predictions = self._evaluate_attacks(self.test_examples, profiles, method, best_sensor)
            method_summary = {"clean": clean_metrics, "attacks": {}}

            for attack_name, rows in attacked_predictions.items():
                write_jsonl(self.output_dir / f"{method}_test_attack_{attack_name}_predictions.jsonl", rows)
                attacked_metrics = summarize_predictions(rows)
                delta = compare_clean_vs_attack(clean_predictions, rows)
                method_summary["attacks"][attack_name] = {**attacked_metrics, **delta}

            summary["methods"][method] = method_summary

        summary["cache"] = {
            "enabled": self.enable_prediction_cache,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._prediction_cache),
        }

        write_json(self.output_dir / "summary.json", summary)
        return summary