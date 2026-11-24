"""
Augmented Deep URLs Experiment.

Evaluates whether augmenting the training set with independently verified
legitimate deep-path and query URLs mitigates the structural false-alarm
weakness of pure lexical phishing detection, while maintaining strict
registered-domain isolation and using standardized features.

Methodology:
1. Load verified legitimate deep URLs (from data/verified_legitimate/)
   and normalized phishing dataset.
2. Formally audit duplicates, label conflicts, and domain overlap.
3. Combine into a separate experimental dataset:
   data/experiments/augmented_deep_urls/dataset_augmented_experiment.csv.
4. Apply PSL-aware GroupShuffleSplit on registered_domain (80/20, seed 42).
5. Extract standardized features (extract_standardized_features).
6. Train experimental Random Forest model (200 trees, balanced, seed 42).
   Saved to models/experiments/url_rf_augmented.joblib.
7. Evaluate test set:
   - Overall metrics (Accuracy, Phish F1, Legit F1, Macro F1, Confusion Matrix)
   - Dedicated performance on legitimate deep URLs in the test partition
   - Dedicated performance on HTTPS phishing URLs in the test partition
   - Scheme invariance probe tests

Usage:
    python agents/url_agent/experiment_augmented_deep_urls.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
import tldextract

from feature_extractor import extract_standardized_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DATA = (
    PROJECT_ROOT / "data" / "normalized_audit" / "phishing_dataset_normalized.csv"
)
VERIFIED_DEEP_DATA = (
    PROJECT_ROOT / "data" / "verified_legitimate" / "verified_legitimate_deep_urls.csv"
)

EXP_DIR = PROJECT_ROOT / "data" / "experiments" / "augmented_deep_urls"
EXP_MODEL_DIR = PROJECT_ROOT / "models" / "experiments"
EXP_MODEL_PATH = EXP_MODEL_DIR / "url_rf_augmented.joblib"

_extractor = tldextract.TLDExtract(include_psl_private_domains=True)


def get_hostname(url: str) -> str:
    val = str(url).strip()
    norm = val if "://" in val else "http://" + val
    try:
        return (urlsplit(norm).hostname or "").strip(".").lower()
    except Exception:
        return ""


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())

    p_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    p_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    p_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    l_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    l_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    l_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "model_name": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "phishing": {
            "precision": float(p_prec),
            "recall": float(p_rec),
            "f1": float(p_f1),
            "detected": tn,
            "missed_false_positive": fp,
        },
        "legitimate": {
            "precision": float(l_prec),
            "recall": float(l_rec),
            "f1": float(l_f1),
            "passed": tp,
            "blocked_false_alarm": fn,
        },
        "confusion_matrix": {
            "actual_0_pred_0": tn,
            "actual_0_pred_1": fp,
            "actual_1_pred_0": fn,
            "actual_1_pred_1": tp,
        },
    }


def main() -> None:
    print("=" * 80)
    print("      AUGMENTED DEEP URLS EXPERIMENT & EVALUATION")
    print("=" * 80)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    EXP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not VERIFIED_DEEP_DATA.exists():
        raise FileNotFoundError(f"Verified dataset missing: {VERIFIED_DEEP_DATA}")

    # --- 1. Load Datasets ---
    print("\n[1/6] Loading source datasets...")
    norm_df = pd.read_csv(
        NORMALIZED_DATA,
        usecols=["URL", "label", "registered_domain"],
        low_memory=False,
    )
    norm_df = norm_df.dropna(subset=["URL", "label"]).copy()
    norm_df["URL"] = norm_df["URL"].astype(str).str.strip()
    norm_df["label"] = pd.to_numeric(norm_df["label"], errors="coerce").astype(int)
    norm_df["hostname"] = norm_df["URL"].map(get_hostname)
    norm_df["registered_domain"] = (
        norm_df["registered_domain"]
        .fillna(norm_df["hostname"])
        .astype(str)
        .str.strip()
        .str.lower()
    )
    norm_df["registered_domain"] = (
        norm_df["registered_domain"]
        .replace({"": None, "nan": None, "none": None})
        .fillna(norm_df["hostname"])
        .astype(str)
    )

    verified_df = pd.read_csv(VERIFIED_DEEP_DATA)
    verified_df["URL"] = verified_df["raw_url"].astype(str).str.strip()
    verified_df["label"] = 1
    verified_df["hostname"] = verified_df["URL"].map(get_hostname)
    verified_df["registered_domain"] = (
        verified_df["registered_domain"].astype(str).str.strip().str.lower()
    )

    print(f"  Existing normalized dataset : {len(norm_df):,} rows")
    print(f"  Verified legitimate deep set: {len(verified_df):,} rows")

    # --- 2. Check Duplicates, Label Conflicts & Domain Overlap ---
    print("\n[2/6] Auditing duplicates and conflicts before combination...")

    # Exact URL duplicates
    norm_urls = set(norm_df["URL"])
    verified_urls = set(verified_df["URL"])
    exact_duplicates = norm_urls.intersection(verified_urls)
    print(f"  Exact URL overlap count : {len(exact_duplicates)}")

    # Check for label conflicts
    if exact_duplicates:
        conflicts = norm_df[norm_df["URL"].isin(exact_duplicates)]
        phishing_conflicts = conflicts[conflicts["label"] == 0]
        if not phishing_conflicts.empty:
            raise RuntimeError(
                f"FATAL: Label conflict! Verified URL was labeled phishing in existing data: {phishing_conflicts['URL'].tolist()}"
            )
        print("  Overlapping URLs are already labeled legitimate. Deduplicating.")
        verified_df = verified_df[~verified_df["URL"].isin(exact_duplicates)].copy()

    # Registered domain audit
    norm_domains = set(norm_df["registered_domain"])
    verified_domains = set(verified_df["registered_domain"])
    shared_domains = norm_domains.intersection(verified_domains)
    print(f"  Shared registered domains count: {len(shared_domains)} {sorted(shared_domains)}")

    # Verify no label conflicts in shared domains
    for d in shared_domains:
        existing_labels = norm_df[norm_df["registered_domain"] == d]["label"].unique()
        print(f"    - Domain '{d}': existing labels = {existing_labels.tolist()} (Expected [1])")
        if 0 in existing_labels:
            print(f"      WARNING: Existing domain '{d}' has phishing rows. Auditing carefully.")

    # Combine datasets
    verified_subset = verified_df[["URL", "label", "registered_domain", "hostname"]].copy()
    verified_subset["is_augmented"] = True
    norm_df["is_augmented"] = False

    combined_df = pd.concat([norm_df, verified_subset], ignore_index=True)
    combined_csv_path = EXP_DIR / "dataset_augmented_experiment.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"  Augmented experimental dataset saved to: {combined_csv_path}")
    print(f"  Total combined rows: {len(combined_df):,}")

    # --- 3. Domain-Separated Train/Test Split ---
    print("\n[3/6] Creating registered-domain-separated split (80/20, seed 42)...")
    groups = combined_df["registered_domain"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(combined_df, combined_df["label"], groups=groups))

    train_domains = set(groups.iloc[train_idx])
    test_domains = set(groups.iloc[test_idx])
    overlap = train_domains & test_domains

    print(f"  Train set: {len(train_idx):,} rows ({len(train_domains):,} domains)")
    print(f"  Test set : {len(test_idx):,} rows ({len(test_domains):,} domains)")
    print(f"  Domain overlap check: {len(overlap)} (MUST be 0)")
    assert len(overlap) == 0, "FATAL: Domain leakage detected!"

    # Distribution of augmented deep URLs across train/test
    aug_train = combined_df.iloc[train_idx]["is_augmented"].sum()
    aug_test = combined_df.iloc[test_idx]["is_augmented"].sum()
    print(f"  Augmented legitimate URLs in Train: {aug_train:,}")
    print(f"  Augmented legitimate URLs in Test : {aug_test:,}")

    # --- 4. Feature Extraction (Standardized) ---
    print("\n[4/6] Extracting standardized features...")
    features_list = [extract_standardized_features(u) for u in combined_df["URL"]]
    X_std = pd.DataFrame(features_list, index=combined_df.index)
    y = combined_df["label"]
    feature_names = X_std.columns.tolist()

    X_train, X_test = X_std.iloc[train_idx], X_std.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- 5. Train Model ---
    print("\n[5/6] Training experimental Random Forest model (200 estimators, balanced)...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "label_values": [0, 1],
            "description": "Experimental Random Forest trained on augmented legitimate deep URLs with standardized features",
        },
        EXP_MODEL_PATH,
    )
    print(f"  Experimental model saved: {EXP_MODEL_PATH}")

    # --- 6. Detailed Evaluation ---
    print("\n[6/6] Evaluating test set and cohort performance...")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, "Augmented Standardized Model")

    # Evaluate specific subsets of the test set:
    test_eval_df = combined_df.iloc[test_idx].copy()
    test_eval_df["y_pred"] = y_pred
    test_eval_df["phish_score"] = y_scores[:, 0] * 100

    def parse_test_structure(u: str) -> tuple[str, bool, bool]:
        norm = u if "://" in u else "http://" + u
        p = urlsplit(norm)
        return p.scheme.lower(), bool((p.path or "").strip("/")), bool((p.query or "").strip())

    t_struct = [parse_test_structure(u) for u in test_eval_df["URL"]]
    test_eval_df["scheme"] = [s[0] for s in t_struct]
    test_eval_df["has_path"] = [s[1] for s in t_struct]
    test_eval_df["has_query"] = [s[2] for s in t_struct]

    # Subset 1: Augmented Legitimate Deep URLs in Test Set
    legit_deep_test = test_eval_df[(test_eval_df["label"] == 1) & (test_eval_df["has_path"])]
    legit_deep_acc = (legit_deep_test["y_pred"] == 1).mean() if len(legit_deep_test) > 0 else 0.0

    # Subset 2: HTTPS Phishing URLs in Test Set
    https_phish_test = test_eval_df[(test_eval_df["label"] == 0) & (test_eval_df["scheme"] == "https")]
    https_phish_caught = (https_phish_test["y_pred"] == 0).sum()
    https_phish_recall = (https_phish_test["y_pred"] == 0).mean() if len(https_phish_test) > 0 else 0.0

    # Subset 3: Legitimate Root URLs in Test Set
    legit_root_test = test_eval_df[(test_eval_df["label"] == 1) & (~test_eval_df["has_path"])]
    legit_root_acc = (legit_root_test["y_pred"] == 1).mean() if len(legit_root_test) > 0 else 0.0

    # Subset 4: Phishing with Deep Paths in Test Set
    phish_deep_test = test_eval_df[(test_eval_df["label"] == 0) & (test_eval_df["has_path"])]
    phish_deep_recall = (phish_deep_test["y_pred"] == 0).mean() if len(phish_deep_test) > 0 else 0.0

    print("\n" + "=" * 80)
    print("               AUGMENTED MODEL OVERALL METRICS")
    print("=" * 80)
    print(f"  Accuracy                : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score          : {metrics['macro_f1']*100:.2f}%")
    print(f"  Phishing Precision (0)  : {metrics['phishing']['precision']*100:.2f}%")
    print(f"  Phishing Recall (0)     : {metrics['phishing']['recall']*100:.2f}%")
    print(f"  Phishing F1-Score (0)   : {metrics['phishing']['f1']*100:.2f}%")
    print(f"  Legitimate Precision (1): {metrics['legitimate']['precision']*100:.2f}%")
    print(f"  Legitimate Recall (1)   : {metrics['legitimate']['recall']*100:.2f}%")
    print(f"  Legitimate F1-Score (1) : {metrics['legitimate']['f1']*100:.2f}%")
    print("-" * 80)
    print("  Confusion Matrix:")
    print(f"    - True Phishing Caught (0->0) : {metrics['phishing']['detected']:,}")
    print(f"    - Missed Phishing (0->1, FP)  : {metrics['phishing']['missed_false_positive']:,}")
    print(f"    - False Alarms (1->0, FN)     : {metrics['legitimate']['blocked_false_alarm']:,}")
    print(f"    - True Legitimate Passed (1->1): {metrics['legitimate']['passed']:,}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("          TARGETED COHORT EVALUATION IN TEST PARTITION")
    print("=" * 80)
    print(f"  Legitimate Deep URLs in Test Set    : {len(legit_deep_test):,} URLs")
    print(f"    - Correctly Passed as Legit (1)   : {(legit_deep_test['y_pred'] == 1).sum():,} ({legit_deep_acc*100:.2f}%)")
    print(f"    - Falsely Blocked as Phish (0)    : {(legit_deep_test['y_pred'] == 0).sum():,} ({(1-legit_deep_acc)*100:.2f}%)")
    print("-" * 80)
    print(f"  HTTPS Phishing URLs in Test Set     : {len(https_phish_test):,} URLs")
    print(f"    - Caught as Phishing (0)          : {https_phish_caught:,} ({https_phish_recall*100:.2f}%)")
    print(f"    - Missed as Legit (1)             : {len(https_phish_test) - https_phish_caught:,} ({(1-https_phish_recall)*100:.2f}%)")
    print("-" * 80)
    print(f"  Legitimate Root URLs in Test Set    : {len(legit_root_test):,} URLs")
    print(f"    - Correctly Passed (1)            : {(legit_root_test['y_pred'] == 1).sum():,} ({legit_root_acc*100:.2f}%)")
    print("-" * 80)
    print(f"  Phishing Deep Path URLs in Test Set : {len(phish_deep_test):,} URLs")
    print(f"    - Caught as Phishing (0)          : {(phish_deep_test['y_pred'] == 0).sum():,} ({phish_deep_recall*100:.2f}%)")
    print("=" * 80)

    # Representative probes
    print("\nDiagnostic Probe URLs on Retrained Augmented Model:")
    probes = [
        ("https://docs.python.org/3/library/json.html", "Official Python docs deep path"),
        ("https://github.com/search?q=python", "Legitimate query URL"),
        ("https://en.wikipedia.org/wiki/Phishing", "Wikipedia legitimate article path"),
        ("http://neverssl.com", "Plain-HTTP benign site"),
        ("https://www.google.com", "Search engine HTTPS root"),
        ("http://www.google.com", "Search engine HTTP root"),
        ("http://192.168.1.1/login/verify.php", "IP credential harvester (HTTP)"),
        ("https://192.168.1.1/login/verify.php", "IP credential harvester (HTTPS)"),
        ("https://gateway.pinata.cloud/ipfs/bafybei.../login.html", "IPFS phishing hash"),
    ]

    probe_evals = []
    for u, desc in probes:
        feat = pd.DataFrame([extract_standardized_features(u)])[feature_names]
        p = model.predict(feat)[0]
        score = model.predict_proba(feat)[0][0] * 100
        verdict = "Phish (0)" if p == 0 else "Legit (1)"
        print(f"  {desc:<38} : {u:<45} -> {verdict:<10} (Phish score: {score:.1f}%)")
        probe_evals.append({
            "url": u,
            "description": desc,
            "prediction": int(p),
            "phish_score": float(score),
        })

    # Save results
    results_json_path = EXP_DIR / "experiment_results.json"
    results_payload = {
        "dataset_rows": len(combined_df),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_domains": len(train_domains),
        "test_domains": len(test_domains),
        "domain_overlap": len(overlap),
        "metrics": metrics,
        "cohort_metrics": {
            "legitimate_deep_total": len(legit_deep_test),
            "legitimate_deep_accuracy": float(legit_deep_acc),
            "https_phishing_total": len(https_phish_test),
            "https_phishing_recall": float(https_phish_recall),
            "legitimate_root_total": len(legit_root_test),
            "legitimate_root_accuracy": float(legit_root_acc),
        },
        "probe_evals": probe_evals,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    report_txt_path = EXP_DIR / "experiment_report.txt"
    report_lines = [
        "AUGMENTED DEEP URLS EXPERIMENT REPORT",
        "=====================================",
        f"Combined dataset records: {len(combined_df):,}",
        f"Train / Test split: {len(train_idx):,} / {len(test_idx):,} (0 domain overlap)",
        "",
        "OVERALL PERFORMANCE:",
        f"  Accuracy       : {metrics['accuracy']*100:.2f}%",
        f"  Macro F1       : {metrics['macro_f1']*100:.2f}%",
        f"  Phish Precision: {metrics['phishing']['precision']*100:.2f}%",
        f"  Phish Recall   : {metrics['phishing']['recall']*100:.2f}%",
        f"  Legit Precision: {metrics['legitimate']['precision']*100:.2f}%",
        f"  Legit Recall   : {metrics['legitimate']['recall']*100:.2f}%",
        "",
        "COHORT PERFORMANCE:",
        f"  Legit Deep URLs in Test     : {len(legit_deep_test)} total -> {legit_deep_acc*100:.2f}% accuracy",
        f"  HTTPS Phishing in Test      : {len(https_phish_test)} total -> {https_phish_recall*100:.2f}% recall",
        f"  Legit Root URLs in Test     : {len(legit_root_test)} total -> {legit_root_acc*100:.2f}% accuracy",
        "",
        "CONFUSION MATRIX:",
        f"  True Phish Caught  : {metrics['phishing']['detected']:,}",
        f"  Missed Phish (FP)  : {metrics['phishing']['missed_false_positive']:,}",
        f"  False Alarms (FN)  : {metrics['legitimate']['blocked_false_alarm']:,}",
        f"  True Legit Passed  : {metrics['legitimate']['passed']:,}",
    ]
    report_txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results JSON  : {results_json_path}")
    print(f"Results Report: {report_txt_path}")
    print(f"Model Artifact: {EXP_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
