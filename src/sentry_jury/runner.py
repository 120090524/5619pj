# from __future__ import annotations

# from pathlib import Path
# from typing import Any
# import hashlib
# import json
# import yaml
# from tqdm import tqdm

# from .aggregator import (
#     aggregate_clean_weighted,
#     aggregate_majority,
#     aggregate_selectively,
#     aggregate_single_sensor,
# )
# from .attacks import Attack, build_attacks
# from .datasets import load_examples, split_calibration_test
# from .metrics import compare_clean_vs_attack, summarize_predictions
# from .profiles import compute_profiles
# from .probes import Probe, build_probes
# from .types import EvalExample, Sensor
# from .utils import ensure_dir, write_json, write_jsonl
# from .judges import build_judge


# class ExperimentRunner:
#     def __init__(self, config: dict[str, Any]) -> None:
#         self.config = config
#         self.seed = int(config.get("seed", 42))
#         self.output_dir = ensure_dir(config.get("output_dir", "runs/default"))
#         self.examples = load_examples(config["dataset"]["path"])
#         self.calibration_examples, self.test_examples = split_calibration_test(
#             self.examples,
#             calibration_fraction=float(config["dataset"].get("calibration_fraction", 0.5)),
#             seed=self.seed,
#         )

#         self.prompt_names: list[str] = list(config["prompts"])
#         self.judges = {spec["name"]: build_judge(spec) for spec in config["judges"]}
#         self.attacks: list[Attack] = build_attacks(list(config.get("attacks", [])))
#         self.probes: list[Probe] = build_probes(list(config.get("probes", [])))
#         self.sensors: list[Sensor] = self._build_sensors()

#         self.lambda_attack = float(config["aggregation"]["lambda_attack"])
#         self.rho_dependence = float(config["aggregation"]["rho_dependence"])
#         self.margin = float(config["aggregation"]["margin"])

#     @classmethod
#     def from_yaml(cls, path: str | Path) -> "ExperimentRunner":
#         with Path(path).open("r", encoding="utf-8") as f:
#             config = yaml.safe_load(f)
#         return cls(config)

#     def _build_sensors(self) -> list[Sensor]:
#         sensors: list[Sensor] = []
#         for judge_name in self.judges:
#             for prompt_name in self.prompt_names:
#                 sensors.append(Sensor(judge_name=judge_name, prompt_name=prompt_name))
#         return sensors

#     def _apply_order_variant(self, example: EvalExample, sensor: Sensor) -> tuple[EvalExample, int]:
#         if sensor.order_variant == "original":
#             return example, 1
#         if sensor.order_variant == "swapped":
#             if example.task_type != "pairwise":
#                 return example, 1
#             if example.response_b is None:
#                 raise ValueError("pairwise example missing response_b")
#             from dataclasses import replace
#             return replace(example, response_a=example.response_b, response_b=example.response_a), -1
#         raise ValueError(f"Unknown order variant: {sensor.order_variant}")

#     def _sensor_predict(self, example: EvalExample, sensor: Sensor) -> int:
#         judge = self.judges[sensor.judge_name]
#         ordered_example, sign_adjustment = self._apply_order_variant(example, sensor)
#         result = judge.predict(ordered_example, sensor.prompt_name)
#         decision = int(result.decision)
#         if decision not in {-1, 1}:
#             raise ValueError(f"Judge returned invalid decision {decision} for sensor {sensor.key}")
#         return sign_adjustment * decision

#     def _collect_clean_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
#         rows: list[dict[str, Any]] = []
#         for example in tqdm(examples, desc="clean calibration", leave=False):
#             for sensor in self.sensors:
#                 pred = self._sensor_predict(example, sensor)
#                 rows.append({
#                     "example_id": example.example_id,
#                     "sensor": sensor.key,
#                     "prediction": pred,
#                     "label": example.label,
#                 })
#         return rows

#     def _collect_probe_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
#         rows: list[dict[str, Any]] = []
#         for example in tqdm(examples, desc="probe calibration", leave=False):
#             for probe in self.probes:
#                 probe_result = probe.apply(example)
#                 for sensor in self.sensors:
#                     pred = self._sensor_predict(probe_result.example, sensor)
#                     rows.append({
#                         "example_id": example.example_id,
#                         "sensor": sensor.key,
#                         "probe_name": probe.name,
#                         "family": probe.family,
#                         "kind": probe.kind,
#                         "prediction": pred,
#                         "label": example.label,
#                     })
#         return rows

#     def _collect_attack_rows(self, examples: list[EvalExample]) -> list[dict[str, Any]]:
#         rows: list[dict[str, Any]] = []
#         for example in tqdm(examples, desc="attack calibration", leave=False):
#             for attack in self.attacks:
#                 attacked = attack.apply(example)
#                 for sensor in self.sensors:
#                     pred = self._sensor_predict(attacked.example, sensor)
#                     rows.append({
#                         "example_id": example.example_id,
#                         "sensor": sensor.key,
#                         "attack_name": attack.name,
#                         "family": attack.family,
#                         "prediction": pred,
#                         "label": example.label,
#                     })
#         return rows

#     def calibrate(self) -> dict[str, dict[str, Any]]:
#         import pandas as pd

#         clean_rows = self._collect_clean_rows(self.calibration_examples)
#         probe_rows = self._collect_probe_rows(self.calibration_examples)
#         attack_rows = self._collect_attack_rows(self.calibration_examples)

#         clean_df = pd.DataFrame(clean_rows)
#         probe_df = pd.DataFrame(probe_rows)
#         attack_df = pd.DataFrame(attack_rows)

#         profiles = compute_profiles(clean_df=clean_df, probe_df=probe_df, attack_df=attack_df)

#         write_jsonl(self.output_dir / "calibration_clean_rows.jsonl", clean_rows)
#         write_jsonl(self.output_dir / "calibration_probe_rows.jsonl", probe_rows)
#         write_jsonl(self.output_dir / "calibration_attack_rows.jsonl", attack_rows)
#         write_json(self.output_dir / "profiles.json", profiles)

#         return profiles

#     def _collect_votes(self, example: EvalExample) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
#         base_votes: dict[str, int] = {}
#         probe_votes_by_family: dict[str, dict[str, int]] = {}

#         for sensor in self.sensors:
#             base_votes[sensor.key] = self._sensor_predict(example, sensor)

#         for probe in self.probes:
#             probe_result = probe.apply(example)
#             family_votes: dict[str, int] = {}
#             for sensor in self.sensors:
#                 family_votes[sensor.key] = self._sensor_predict(probe_result.example, sensor)
#             probe_votes_by_family[probe.family] = family_votes

#         return base_votes, probe_votes_by_family

#     def _predict_example_with_method(
#         self,
#         example: EvalExample,
#         profiles: dict[str, dict[str, Any]],
#         method: str,
#         best_sensor: str,
#     ) -> dict[str, Any]:
#         base_votes, probe_votes_by_family = self._collect_votes(example)

#         if method == "single_best":
#             agg = aggregate_single_sensor(base_votes, best_sensor)
#         elif method == "majority_vote":
#             agg = aggregate_majority(base_votes)
#         elif method == "clean_weighted":
#             agg = aggregate_clean_weighted(base_votes, profiles, margin=self.margin)
#         elif method == "sentry":
#             agg = aggregate_selectively(
#                 base_votes=base_votes,
#                 profiles=profiles,
#                 probe_votes_by_family=probe_votes_by_family,
#                 lambda_attack=self.lambda_attack,
#                 rho_dependence=self.rho_dependence,
#                 margin=self.margin,
#             )
#         else:
#             raise ValueError(f"Unknown method: {method}")

#         return {
#             "example_id": example.example_id,
#             "label": example.label,
#             "prediction": agg.prediction,
#             "abstained": agg.abstained,
#             "score": agg.score,
#             "confidence": agg.confidence,
#             "instance_risks": agg.instance_risks,
#             "sensor_weights": agg.sensor_weights,
#         }

#     def _evaluate_split(self, examples: list[EvalExample], profiles: dict[str, dict[str, Any]], method: str, best_sensor: str) -> list[dict[str, Any]]:
#         rows: list[dict[str, Any]] = []
#         for example in tqdm(examples, desc=f"{method}", leave=False):
#             rows.append(self._predict_example_with_method(example, profiles, method, best_sensor))
#         return rows

#     def _evaluate_attacks(
#         self,
#         examples: list[EvalExample],
#         profiles: dict[str, dict[str, Any]],
#         method: str,
#         best_sensor: str,
#     ) -> dict[str, list[dict[str, Any]]]:
#         outputs: dict[str, list[dict[str, Any]]] = {}
#         for attack in self.attacks:
#             rows: list[dict[str, Any]] = []
#             for example in tqdm(examples, desc=f"{method}:{attack.name}", leave=False):
#                 attacked = attack.apply(example)
#                 pred = self._predict_example_with_method(attacked.example, profiles, method, best_sensor)
#                 pred["attack_name"] = attack.name
#                 rows.append(pred)
#             outputs[attack.name] = rows
#         return outputs

#     def run(self) -> dict[str, Any]:
#         profiles = self.calibrate()
#         best_sensor = max(profiles.items(), key=lambda kv: kv[1]["base_weight"])[0]

#         methods = ["single_best", "majority_vote", "clean_weighted", "sentry"]
#         summary = {
#             "project_name": self.config.get("project_name", "sentry_jury"),
#             "num_examples_total": len(self.examples),
#             "num_examples_calibration": len(self.calibration_examples),
#             "num_examples_test": len(self.test_examples),
#             "num_sensors": len(self.sensors),
#             "best_sensor": best_sensor,
#             "methods": {},
#         }

#         for method in methods:
#             clean_predictions = self._evaluate_split(self.test_examples, profiles, method, best_sensor)
#             write_jsonl(self.output_dir / f"{method}_test_clean_predictions.jsonl", clean_predictions)
#             clean_metrics = summarize_predictions(clean_predictions)

#             attacked_predictions = self._evaluate_attacks(self.test_examples, profiles, method, best_sensor)
#             method_summary = {"clean": clean_metrics, "attacks": {}}

#             for attack_name, rows in attacked_predictions.items():
#                 write_jsonl(self.output_dir / f"{method}_test_attack_{attack_name}_predictions.jsonl", rows)
#                 attacked_metrics = summarize_predictions(rows)
#                 delta = compare_clean_vs_attack(clean_predictions, rows)
#                 method_summary["attacks"][attack_name] = {**attacked_metrics, **delta}

#             summary["methods"][method] = method_summary

#         write_json(self.output_dir / "summary.json", summary)
#         return summary

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
from .metrics import compare_clean_vs_attack, summarize_predictions
from .profiles import compute_profiles
from .probes import Probe, build_probes
from .types import EvalExample, Sensor
from .utils import ensure_dir, write_json, write_jsonl
from .judges import build_judge


class ExperimentRunner:
    """Run calibration + clean evaluation + attacked evaluation.

    Backward compatible with the original course-project runner, while adding:
    1. dataset.calibration_path / dataset.test_path support
    2. optional calibration_limit / test_limit
    3. in-memory prediction caching to avoid repeated API calls
    4. configurable methods list in YAML
    5. configurable order_variants for pairwise experiments
    6. configurable attacks_target_field (default: response_a)
    7. safe handling when probes or attacks are empty
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        self.output_dir = ensure_dir(config.get("output_dir", "runs/default"))

        dataset_cfg = config["dataset"]
        self.examples, self.calibration_examples, self.test_examples = self._load_dataset_splits(dataset_cfg)

        # Optional debug limits.
        calibration_limit = dataset_cfg.get("calibration_limit")
        test_limit = dataset_cfg.get("test_limit")
        if calibration_limit is not None:
            self.calibration_examples = self.calibration_examples[: int(calibration_limit)]
        if test_limit is not None:
            self.test_examples = self.test_examples[: int(test_limit)]

        # Keep examples as the union actually used in this run.
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

        # Disk-backed cache: (sensor_key, content_hash) -> (decision, category, severity).
        self.enable_prediction_cache = bool(config.get("enable_prediction_cache", True))
        self._disk_cache_path = self.output_dir / "prediction_cache.json"
        self._prediction_cache: dict[tuple[str, str], tuple[int, str, int]] = self._load_disk_cache()
        self._cache_hits = 0
        self._cache_misses = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentRunner":
        with Path(path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _load_dataset_splits(
        self,
        dataset_cfg: dict[str, Any],
    ) -> tuple[list[EvalExample], list[EvalExample], list[EvalExample]]:
        """Support either:
        A) old style: dataset.path + calibration_fraction
        B) new style: dataset.calibration_path + dataset.test_path
        """
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

    def _load_disk_cache(self) -> dict[tuple[str, str], tuple[int, str, int]]:
        if not self.enable_prediction_cache or not self._disk_cache_path.exists():
            return {}
        try:
            with self._disk_cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded: dict[tuple[str, str], tuple[int, str, int]] = {}
            for k, v in raw.items():
                key = tuple(k.split("|||", 1))
                if isinstance(v, list) and len(v) == 3:
                    loaded[key] = (int(v[0]), str(v[1]), int(v[2]))
                else:
                    # Legacy cache (decision-only). Reuse decision, mark category unknown.
                    decision = int(v)
                    category = "none" if decision == -1 else "unknown"
                    loaded[key] = (decision, category, 0)
            print(f"[cache] Loaded {len(loaded)} entries from disk cache.")
            return loaded
        except Exception:
            return {}

    def _save_disk_cache(self) -> None:
        if not self.enable_prediction_cache:
            return
        raw = {"|||".join(k): list(v) for k, v in self._prediction_cache.items()}
        with self._disk_cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f)

    def _example_cache_key(self, example: EvalExample) -> str:
        payload = example.to_dict()
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _sensor_predict(self, example: EvalExample, sensor: Sensor) -> tuple[int, str, int]:
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
        category = getattr(result, "category", "unknown") or "unknown"
        severity = int(getattr(result, "severity", 0) or 0)
        record = (final_decision, category, severity)
        if self.enable_prediction_cache:
            self._prediction_cache[cache_key] = record
            self._save_disk_cache()
        return record

    def _collect_clean_rows(
        self, examples: list[EvalExample]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="clean calibration", leave=False):
            for sensor in self.sensors:
                pred, category, severity = self._sensor_predict(example, sensor)
                rows.append(
                    {
                        "example_id": example.example_id,
                        "sensor": sensor.key,
                        "prediction": pred,
                        "label": example.label,
                    }
                )
                category_rows.append(
                    {
                        "example_id": example.example_id,
                        "sensor": sensor.key,
                        "prediction": pred,
                        "label": example.label,
                        "category": category,
                        "severity": severity,
                    }
                )
        return rows, category_rows

    def _collect_probe_rows(
        self, examples: list[EvalExample]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="probe calibration", leave=False):
            for probe in self.probes:
                probe_result = probe.apply(example)
                for sensor in self.sensors:
                    pred, category, severity = self._sensor_predict(probe_result.example, sensor)
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "sensor": sensor.key,
                            "probe_name": probe.name,
                            "family": probe.family,
                            "kind": probe.kind,
                            "prediction": pred,
                            "label": example.label,
                        }
                    )
                    category_rows.append(
                        {
                            "example_id": example.example_id,
                            "sensor": sensor.key,
                            "probe_name": probe.name,
                            "family": probe.family,
                            "kind": probe.kind,
                            "prediction": pred,
                            "label": example.label,
                            "category": category,
                            "severity": severity,
                        }
                    )
        return rows, category_rows

    def _collect_attack_rows(
        self, examples: list[EvalExample]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc="attack calibration", leave=False):
            for attack in self.attacks:
                attacked = attack.apply(example)
                for sensor in self.sensors:
                    pred, category, severity = self._sensor_predict(attacked.example, sensor)
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "sensor": sensor.key,
                            "attack_name": attack.name,
                            "family": attack.family,
                            "prediction": pred,
                            "label": example.label,
                        }
                    )
                    category_rows.append(
                        {
                            "example_id": example.example_id,
                            "sensor": sensor.key,
                            "attack_name": attack.name,
                            "family": attack.family,
                            "prediction": pred,
                            "label": example.label,
                            "category": category,
                            "severity": severity,
                        }
                    )
        return rows, category_rows

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

        clean_rows, clean_cat_rows = self._collect_clean_rows(self.calibration_examples)
        probe_rows, probe_cat_rows = self._collect_probe_rows(self.calibration_examples)
        attack_rows, attack_cat_rows = self._collect_attack_rows(self.calibration_examples)

        clean_df = pd.DataFrame(clean_rows) if clean_rows else self._empty_clean_df()
        probe_df = pd.DataFrame(probe_rows) if probe_rows else self._empty_probe_df()
        attack_df = pd.DataFrame(attack_rows) if attack_rows else self._empty_attack_df()

        profiles = compute_profiles(clean_df=clean_df, probe_df=probe_df, attack_df=attack_df)

        write_jsonl(self.output_dir / "calibration_clean_rows.jsonl", clean_rows)
        write_jsonl(self.output_dir / "calibration_probe_rows.jsonl", probe_rows)
        write_jsonl(self.output_dir / "calibration_attack_rows.jsonl", attack_rows)
        write_jsonl(self.output_dir / "category_clean_rows.jsonl", clean_cat_rows)
        write_jsonl(self.output_dir / "category_probe_rows.jsonl", probe_cat_rows)
        write_jsonl(self.output_dir / "category_attack_rows.jsonl", attack_cat_rows)
        write_json(self.output_dir / "profiles.json", profiles)

        return profiles

    def _collect_votes(
        self, example: EvalExample
    ) -> tuple[
        dict[str, int],
        dict[str, dict[str, int]],
        dict[str, tuple[str, int]],
    ]:
        base_votes: dict[str, int] = {}
        base_extras: dict[str, tuple[str, int]] = {}
        probe_votes_by_family: dict[str, dict[str, int]] = {}

        for sensor in self.sensors:
            decision, category, severity = self._sensor_predict(example, sensor)
            base_votes[sensor.key] = decision
            base_extras[sensor.key] = (category, severity)

        for probe in self.probes:
            probe_result = probe.apply(example)
            family_votes: dict[str, int] = {}
            for sensor in self.sensors:
                decision, _category, _severity = self._sensor_predict(probe_result.example, sensor)
                family_votes[sensor.key] = decision
            probe_votes_by_family[probe.family] = family_votes

        return base_votes, probe_votes_by_family, base_extras

    def _predict_example_with_method(
        self,
        example: EvalExample,
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> tuple[dict[str, Any], dict[str, tuple[str, int]]]:
        base_votes, probe_votes_by_family, base_extras = self._collect_votes(example)

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
        else:
            raise ValueError(f"Unknown method: {method}")

        prediction_row = {
            "example_id": example.example_id,
            "label": example.label,
            "prediction": agg.prediction,
            "abstained": agg.abstained,
            "score": agg.score,
            "confidence": agg.confidence,
            "instance_risks": agg.instance_risks,
            "sensor_weights": agg.sensor_weights,
        }
        return prediction_row, base_extras

    def _evaluate_split(
        self,
        examples: list[EvalExample],
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        for example in tqdm(examples, desc=f"{method}", leave=False):
            pred_row, extras = self._predict_example_with_method(example, profiles, method, best_sensor)
            rows.append(pred_row)
            for sensor_key, (category, severity) in extras.items():
                category_rows.append(
                    {
                        "example_id": example.example_id,
                        "sensor": sensor_key,
                        "label": example.label,
                        "category": category,
                        "severity": severity,
                    }
                )
        return rows, category_rows

    def _evaluate_attacks(
        self,
        examples: list[EvalExample],
        profiles: dict[str, dict[str, Any]],
        method: str,
        best_sensor: str,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        category_outputs: dict[str, list[dict[str, Any]]] = {}
        for attack in self.attacks:
            rows: list[dict[str, Any]] = []
            cat_rows: list[dict[str, Any]] = []
            for example in tqdm(examples, desc=f"{method}:{attack.name}", leave=False):
                attacked = attack.apply(example)
                pred_row, extras = self._predict_example_with_method(
                    attacked.example, profiles, method, best_sensor
                )
                pred_row["attack_name"] = attack.name
                rows.append(pred_row)
                for sensor_key, (category, severity) in extras.items():
                    cat_rows.append(
                        {
                            "example_id": example.example_id,
                            "sensor": sensor_key,
                            "label": example.label,
                            "attack_name": attack.name,
                            "family": attack.family,
                            "category": category,
                            "severity": severity,
                        }
                    )
            outputs[attack.name] = rows
            category_outputs[attack.name] = cat_rows
        return outputs, category_outputs

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
            clean_predictions, clean_category_rows = self._evaluate_split(
                self.test_examples, profiles, method, best_sensor
            )
            write_jsonl(self.output_dir / f"{method}_test_clean_predictions.jsonl", clean_predictions)
            if method == self.methods[0]:
                # Per-sensor category rows are method-independent (depend on judge output, not aggregation).
                # Write once to avoid 4x duplication.
                write_jsonl(self.output_dir / "category_test_clean_rows.jsonl", clean_category_rows)
            clean_metrics = summarize_predictions(clean_predictions)

            attacked_predictions, attacked_category_rows = self._evaluate_attacks(
                self.test_examples, profiles, method, best_sensor
            )
            method_summary = {"clean": clean_metrics, "attacks": {}}

            for attack_name, rows in attacked_predictions.items():
                write_jsonl(self.output_dir / f"{method}_test_attack_{attack_name}_predictions.jsonl", rows)
                if method == self.methods[0]:
                    write_jsonl(
                        self.output_dir / f"category_test_attack_{attack_name}_rows.jsonl",
                        attacked_category_rows[attack_name],
                    )
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