#!/usr/bin/env python3
"""Build an offline, multi-day NoMAD results report from evaluation JSON files."""
import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path


BUNDLE_FORMAT = "nomad-results-bundle"
BUNDLE_VERSION = 1
CONDITION_LABELS = {
    "unchanged": "Unchanged Day-0 transfer",
    "normalization_only": "Normalization-only (target-training calibrated)",
    "alignment_unsupervised": "NoMAD alignment (unsupervised)",
    "alignment_supervised": "NoMAD alignment (supervised)",
    "day0": "Day-0 reference",
}


def read_json(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON from {path}: {exc}") from exc


def split_condition(label):
    value = str(label).casefold()
    if "unchanged day-0 transfer" in value:
        return "unchanged"
    if "normalization-only" in value:
        return "normalization_only"
    if "unsupervised alignment" in value:
        return "alignment_unsupervised"
    if "supervised alignment" in value:
        return "alignment_supervised"
    if value.startswith("day-0"):
        return "day0"
    return "other:" + str(label)


def _finite_score(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Score for {context} must be a finite number; got {value!r}.")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Score for {context} must be finite; got {value!r}.")
    return score


def normalize_evaluation(report, dayk_label, seed_override=None, source=""):
    if not isinstance(report, dict):
        raise ValueError(f"Evaluation report {source or '(input)'} must be a JSON object.")
    args = report.get("args")
    if not isinstance(args, dict):
        args = {}
    seed = seed_override if seed_override is not None else args.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"A nonnegative run seed is required for {source or dayk_label}; pass @SEED or include integer args.seed.")
    split = report.get("split")
    if split not in ("test", "validation"):
        raise ValueError(f"{source or dayk_label} has invalid split {split!r}; expected test or validation.")
    label = str(dayk_label).strip()
    if not label:
        raise ValueError("Day-K label cannot be empty.")
    target_hash = report.get("target_sha256")
    reference_hash = report.get("reference_sha256")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{source or dayk_label} is missing a results list.")
    rows = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"Result row {index} in {source or dayk_label} must be an object.")
        method = row.get("method")
        split_label = row.get("split")
        scores = row.get("scores")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"Result row {index} in {source or dayk_label} has no method name.")
        if not isinstance(split_label, str) or not isinstance(scores, dict) or not scores:
            raise ValueError(f"Result row {index} in {source or dayk_label} needs split text and scores.")
        normalized_scores = {
            str(metric): _finite_score(value, f"{method}/{split_label}/{metric}")
            for metric, value in scores.items()
        }
        rows.append({
            "method": method.strip(),
            "condition": split_condition(split_label),
            "condition_label": CONDITION_LABELS.get(split_condition(split_label), str(split_label)),
            "split_label": split_label,
            "scores": normalized_scores,
        })
    return {
        "dayk": label,
        "target_sha256": str(target_hash) if target_hash else None,
        "reference_sha256": str(reference_hash) if reference_hash else None,
        "reference_model_id": report.get("reference_model_id"),
        "target_model_id": report.get("target_model_id"),
        "seed": seed,
        "split": split,
        "source": str(source),
        "alignment": report.get("alignment", "unspecified"),
        "trial_counts": report.get("trial_counts", {}),
        "rows": rows,
    }


def _result_argument(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid --result {spec!r}; expected DAYK[@SEED]=PATH.")
    identity, path = spec.split("=", 1)
    identity = identity.strip()
    seed = None
    if "@" in identity:
        maybe_label, maybe_seed = identity.rsplit("@", 1)
        if not maybe_seed.isdecimal():
            raise ValueError(f"Invalid seed in --result {spec!r}; @SEED must be a nonnegative integer.")
        identity, seed = maybe_label.strip(), int(maybe_seed)
    if not identity or not path.strip():
        raise ValueError(f"Invalid --result {spec!r}; both Day-K label and path are required.")
    return identity, seed, Path(path.strip())


def _summary_reports(path):
    summary = read_json(path)
    days = summary.get("days") if isinstance(summary, dict) else None
    if not isinstance(days, list):
        raise ValueError(f"Summary {path} must contain a days list.")
    records = []
    for index, entry in enumerate(days):
        if not isinstance(entry, dict) or not isinstance(entry.get("report"), dict):
            raise ValueError(f"Summary {path} day entry {index} must contain an evaluation report.")
        report = entry["report"]
        label = entry.get("target") or Path(entry.get("archive") or "dayk").stem
        source = entry.get("archive") or f"{path} (Day-K entry {index + 1})"
        records.append(normalize_evaluation(report, label, source=source))
    return records


def _validate_records(records):
    if not records:
        raise ValueError("No evaluation records were loaded.")
    reference_hashes = {r["reference_sha256"] for r in records if r.get("reference_sha256")}
    if len(reference_hashes) > 1:
        raise ValueError("Inputs contain different reference data hashes; use one reference dataset per report.")
    labels_by_identity, identities_by_label = {}, {}
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each normalized result record must be an object.")
        if not isinstance(record.get("dayk"), str) or not record["dayk"].strip():
            raise ValueError("Each normalized result needs a non-empty Day-K label.")
        if (isinstance(record.get("seed"), bool) or not isinstance(record.get("seed"), int) or
                record["seed"] < 0):
            raise ValueError(f"Normalized result for {record.get('dayk')!r} needs a nonnegative integer run seed.")
        if record.get("split") not in ("test", "validation"):
            raise ValueError(f"Normalized result for {record['dayk']!r} has an invalid split.")
        if not isinstance(record.get("rows"), list) or not record["rows"]:
            raise ValueError(f"Normalized result for {record['dayk']!r} needs at least one score row.")
        for row in record["rows"]:
            if (not isinstance(row, dict) or not isinstance(row.get("method"), str) or
                    not isinstance(row.get("condition"), str) or not isinstance(row.get("scores"), dict) or
                    not row["scores"]):
                raise ValueError(f"Normalized result for {record['dayk']!r} contains an invalid score row.")
            for metric, value in row["scores"].items():
                _finite_score(value, f"{record['dayk']}/{row['method']}/{metric}")
        identity = record.get("target_sha256") or f"label:{record['dayk']}"
        previous_label = labels_by_identity.setdefault(identity, record["dayk"])
        if previous_label != record["dayk"]:
            raise ValueError(f"Target identity {identity} has conflicting Day-K labels {previous_label!r} and {record['dayk']!r}.")
        previous_identity = identities_by_label.setdefault(record["dayk"], identity)
        if previous_identity != identity:
            raise ValueError(f"Day-K label {record['dayk']!r} refers to multiple target identities; use distinct labels.")
        key = (identity, record["seed"], record["split"])
        if key in seen:
            raise ValueError(f"Duplicate report for Day-K {record['dayk']!r}, seed {record['seed']}, split {record['split']}.")
        seen.add(key)
    return records


def load_report_inputs(summaries, results, bundles):
    records = []
    for path in summaries:
        records.extend(_summary_reports(path))
    for spec in results:
        label, seed, path = _result_argument(spec)
        report = read_json(path)
        records.append(normalize_evaluation(report, label, seed_override=seed, source=str(path)))
    for path in bundles:
        bundle = read_json(path)
        if (not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT or
                bundle.get("version") != BUNDLE_VERSION or not isinstance(bundle.get("records"), list)):
            raise ValueError(f"Bundle {path} has an unsupported format or version.")
        records.extend(bundle["records"])
    return _validate_records(records)


def _sample_stats(values):
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def aggregate_records(records, split):
    if split not in ("test", "validation"):
        raise ValueError("split must be 'test' or 'validation'.")
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No {split} records are available.")
    day_records = {}
    series_values = {}
    definitions = {}
    for record in selected:
        identity = record.get("target_sha256") or f"label:{record['dayk']}"
        day = day_records.setdefault(identity, {
            "id": identity,
            "label": record["dayk"],
            "target_sha256": record.get("target_sha256"),
            "seeds": set(),
            "records": [],
        })
        day["seeds"].add(record["seed"])
        day["records"].append(record)
        for row in record["rows"]:
            for metric, value in row["scores"].items():
                key = "\u001f".join((row["method"], row["condition"], metric))
                definitions[key] = {
                    "key": key,
                    "method": row["method"],
                    "condition": row["condition"],
                    "condition_label": row["condition_label"],
                    "metric": metric,
                }
                series_values.setdefault((identity, key), {})[record["seed"]] = value
    days = []
    for identity, day in sorted(day_records.items(), key=lambda item: item[1]["label"].casefold()):
        series = {}
        for key in definitions:
            observations = series_values.get((identity, key), {})
            if observations:
                stats = _sample_stats(list(observations.values()))
                stats["seeds"] = [
                    {"seed": seed, "value": value}
                    for seed, value in sorted(observations.items())
                ]
                series[key] = stats
        days.append({
            "id": identity,
            "label": day["label"],
            "target_sha256": day["target_sha256"],
            "seed_count": len(day["seeds"]),
            "seeds": sorted(day["seeds"]),
            "series": series,
            "records": day["records"],
        })
    overall = {}
    for key, definition in definitions.items():
        means = [day["series"][key]["mean"] for day in days if key in day["series"]]
        if means:
            stats = _sample_stats(means)
            stats["day_count"] = len(means)
            stats["seed_count"] = sum(day["series"][key]["n"] for day in days if key in day["series"])
            overall[key] = stats
    return {
        "split": split,
        "days": days,
        "series": list(definitions.values()),
        "overall": overall,
        "record_count": len(selected),
        "reference_sha256": next((r["reference_sha256"] for r in selected if r.get("reference_sha256")), None),
    }


def write_report(records, output_dir):
    output_dir = Path(output_dir)
    assets = Path(__file__).with_name("report")
    if output_dir.resolve() == assets.resolve():
        raise ValueError("Choose an output directory outside the source report/ directory.")
    for name in ("index.html", "styles.css", "app.js", "report-data.js"):
        source = assets / name
        if not source.is_file():
            raise ValueError(f"Required report UI asset is missing: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js", "report-data.js"):
        shutil.copy2(assets / name, output_dir / name)
    _validate_records(records)
    bundle = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION, "records": records}
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (output_dir / "report-data.js").write_text(
        f"window.NOMAD_REPORT_BUNDLE = {payload};\n", encoding="utf-8"
    )
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Generate an offline NoMAD multi-day results report.")
    parser.add_argument("--summary", action="append", type=Path, default=[],
                        help="cross_day_summary.json; repeat once per seed run")
    parser.add_argument("--result", action="append", default=[], metavar="DAYK[@SEED]=PATH",
                        help="individual evaluation JSON; optional seed override")
    parser.add_argument("--bundle", action="append", type=Path, default=[],
                        help="normalized JSON bundle exported by the report")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the static report")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.summary or args.result or args.bundle):
        parser.error("provide at least one --summary, --result, or --bundle")
    try:
        records = load_report_inputs(args.summary, args.result, args.bundle)
        test_records = [record for record in records if record["split"] == "test"]
        validation_records = [record for record in records if record["split"] == "validation"]
        if not test_records and not validation_records:
            raise ValueError("Inputs contain no test or validation reports.")
        write_report(records, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"saved {args.out.resolve()}/index.html")
    print(f"loaded {len(records)} report runs: {len(test_records)} test, {len(validation_records)} validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
