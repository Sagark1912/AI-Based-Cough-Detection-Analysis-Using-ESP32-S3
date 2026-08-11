#!/usr/bin/env python3
"""COUGHVID validation and metadata-cleaning stage only.

This stage deliberately does not read or transform audio. Audio preprocessing,
feature extraction, training, and evaluation consume its output only after the
complete metadata-cleaning report passes.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_LABELS = {
    "cough_type": ["dry", "wet", "unknown"],
    "abnormalities": ["wheezing", "dyspnea", "congestion", "nothing"],
    "diagnosis": ["COVID-19", "healthy_cough", "lower_infection", "upper_infection", "obstructive_disease"],
    "severity": ["mild", "pseudocough", "severe", "unknown"],
    "overall_status": ["healthy", "symptomatic", "COVID-19"],
}
MISSING = {"", "nan", "none", "null", "na", "n/a", "unknown", "missing"}
TRUE = {"true", "1", "yes", "present"}


def norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    names = {norm(c): c for c in frame.columns}
    return next((names[norm(c)] for c in candidates if norm(c) in names), None)


def consensus(row: pd.Series, columns: list[str]) -> tuple[str, int]:
    values = [norm(row[c]) for c in columns if c in row.index and norm(row[c]) not in MISSING]
    if not values:
        return "unknown", 0
    counts = pd.Series(values).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "unknown", 0
    return str(counts.index[0]), int(counts.iloc[0])


def add_single_head(frame: pd.DataFrame, task: str, columns: list[str], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    labels, votes = [], []
    aliases = aliases or {}
    canonical = {norm(label): label for label in OUTPUT_LABELS[task]}
    trainable = {"cough_type": {"dry", "wet"}, "diagnosis": {"covid_19", "healthy_cough", "lower_infection", "upper_infection", "obstructive_disease"}, "severity": {"mild", "pseudocough", "severe"}, "overall_status": {"healthy", "symptomatic", "covid_19"}}.get(task, set(canonical))
    for _, row in frame.iterrows():
        label, vote = consensus(row, columns)
        label_norm = norm(aliases.get(label, label))
        if label_norm not in canonical:
            label, vote = "unknown", 0
        else:
            label = canonical[label_norm]
            if label_norm not in trainable:
                vote = 0
        labels.append(label)
        votes.append(vote)
    frame[f"target_{task}"] = labels
    frame[f"mask_{task}"] = [int(v > 0) for v in votes]
    frame[f"consensus_votes_{task}"] = votes
    return {"reviewer_columns": columns, "trainable_labels": sorted(trainable), "valid_rows": int(sum(v > 0 for v in votes)), "label_counts": pd.Series(labels).value_counts().to_dict(), "masked_unknown_or_unsupported": int(sum((v == 0) for v in votes))}


def add_abnormalities(frame: pd.DataFrame, columns_by_label: dict[str, list[str]]) -> dict[str, Any]:
    labels, masks = [], []
    for _, row in frame.iterrows():
        present, observed = [], 0
        for label, columns in columns_by_label.items():
            values = [norm(row[c]) for c in columns if c in row.index and norm(row[c]) not in MISSING]
            if values:
                observed += 1
                if any(value in TRUE for value in values):
                    present.append(label)
        labels.append("|".join(present) if present else ("nothing" if observed else "unknown"))
        masks.append(int(observed > 0))
    frame["target_abnormalities"] = labels
    frame["mask_abnormalities"] = masks
    frame["consensus_votes_abnormalities"] = masks
    return {"reviewer_columns": sum(columns_by_label.values(), []), "valid_rows": int(sum(masks)), "label_counts": pd.Series(labels).value_counts().to_dict(), "representation": "multi-label pipe-delimited; empty positive set is nothing"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-recording-groups", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.metadata.is_file():
        raise SystemExit(f"Metadata not found: {args.metadata}")
    if not args.audio_root.is_dir():
        raise SystemExit(f"Audio directory not found: {args.audio_root}")
    frame = pd.read_csv(args.metadata) if args.metadata.suffix.lower() == ".csv" else pd.read_json(args.metadata)
    input_fields = len(frame.columns)
    if input_fields < 52:
        raise SystemExit(f"Refusing schema: expected at least 52 fields, found {input_fields}")
    input_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    uuid_col = find_column(frame, ["uuid"])
    if not uuid_col:
        raise SystemExit("UUID field is required for audio matching")
    frame["audio_filename"] = frame[uuid_col].astype(str).map(lambda x: x + ".wav")
    frame["raw_audio_path"] = frame["audio_filename"].map(lambda x: str(args.audio_root / x) if (args.audio_root / x).is_file() else "")
    frame["audio_exists"] = frame["raw_audio_path"].map(lambda x: int(bool(x)))
    cough_cols = [c for c in frame.columns if norm(c).startswith("cough_type_")]
    diagnosis_cols = [c for c in frame.columns if norm(c).startswith("diagnosis_")]
    severity_cols = [c for c in frame.columns if norm(c).startswith("severity_")]
    abnormal_cols = {label: [c for c in frame.columns if norm(c).startswith(label + "_")] for label in ["wheezing", "dyspnea", "congestion", "nothing"]}
    source_status_counts = frame["status"].dropna().astype(str).str.strip().value_counts().to_dict() if "status" in frame.columns else {}
    source_diagnosis_counts = {c: frame[c].dropna().astype(str).str.strip().value_counts().to_dict() for c in diagnosis_cols}
    reports = {"input_rows": input_rows, "output_rows": len(frame), "input_fields": input_fields, "duplicate_rows_removed": duplicate_rows, "audio_files_found": int(frame.audio_exists.sum()), "audio_files_missing": int((frame.audio_exists == 0).sum()), "audio_transformations": "none in cleaning stage", "source_label_counts": {"status": source_status_counts, "diagnosis": source_diagnosis_counts}, "tasks": {}}
    reports["tasks"]["cough_type"] = add_single_head(frame, "cough_type", cough_cols)
    reports["tasks"]["abnormalities"] = add_abnormalities(frame, abnormal_cols)
    reports["tasks"]["diagnosis"] = add_single_head(frame, "diagnosis", diagnosis_cols, {"covid_19": "covid-19"})
    reports["tasks"]["severity"] = add_single_head(frame, "severity", severity_cols)
    status_col = find_column(frame, ["status"])
    reports["tasks"]["overall_status"] = add_single_head(frame, "overall_status", [status_col] if status_col else [], {"covid_19": "covid-19"})
    group_col = next((c for c in frame.columns if norm(c) in {"patient_id", "subject_id", "user_id"}), None)
    if not group_col:
        if not args.allow_recording_groups:
            raise SystemExit("No participant identifier. Use --allow-recording-groups only to record the UUID-level split limitation.")
        group_col = uuid_col
        reports["split_limitation"] = "No participant identifier; UUID recording-level split used and participant leakage cannot be ruled out."
    groups = frame[group_col].fillna("missing_group").astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(frame, groups=groups))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + 1)
    rel_train, val_rel = next(split2.split(train_idx, groups=groups.iloc[train_idx]))
    frame["split"] = "test"
    frame.loc[train_idx[rel_train], "split"] = "train"
    frame.loc[train_idx[val_rel], "split"] = "validation"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    reports.update({"group_column": group_col, "split_counts": frame["split"].value_counts().to_dict(), "physiological_labels": "not present in COUGHVID; acquire independently on hardware", "next_stage": "Only after reviewing this complete cleaning report: audio preprocessing, cough detection/segmentation, feature extraction, training, evaluation."})
    args.out.with_suffix(".report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
