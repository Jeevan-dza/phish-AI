

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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
import tldextract

from feature_extractor import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT / "data" / "normalized_audit" / "phishing_dataset_normalized.csv"
)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"

EXP_MODEL_DIR = PROJECT_ROOT / "models" / "experiments"
EXP_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "experiments" / "https_robustness"

_extractor = tldextract.TLDExtract(include_psl_private_domains=True)


def get_hostname(url: str) -> str:
    value = str(url).strip()
    if "://" not in value:
        value = "http://" + value
    try:
        return (urlsplit(value).hostname or "").strip(".").lower()
    except (ValueError, UnicodeError):
        return ""


def get_registered_domain(url: str) -> str:
    host = get_hostname(url)
    if not host:
        return ""
    try:
        res = _extractor(host)
        rd = res.top_domain_under_public_suffix or res.registered_domain
        return rd.lower() if rd else host
    except Exception:
        return host


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, name: str) -> dict:
    """Compute complete classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # cm format for labels [0, 1]:
    # [[TN, FP],
    #  [FN, TP]]
    # Where:
    # 0 = Phishing (negative class)
    # 1 = Legitimate (positive class)
    # tn: actual 0, pred 0 (phishing correctly detected)
    # fp: actual 0, pred 1 (phishing missed / allowed) -> CRITICAL SECURITY RISK
    # fn: actual 1, pred 0 (legitimate blocked / false alarm) -> USER FRICTION
    # tp: actual 1, pred 1 (legitimate correctly passed)
    tn, fp, fn, tp = cm.ravel()

    # Phishing metrics (treating label 0 as target of interest)
    p_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    p_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    p_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Legitimate metrics (treating label 1 as target)
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
            "detected": int(tn),
            "missed_false_positive": int(fp),
        },
        "legitimate": {
            "precision": float(l_prec),
            "recall": float(l_rec),
            "f1": float(l_f1),
            "passed": int(tp),
            "blocked_false_alarm": int(fn),
        },
        "confusion_matrix": {
            "actual_0_pred_0": int(tn),
            "actual_0_pred_1": int(fp),
            "actual_1_pred_0": int(fn),
            "actual_1_pred_1": int(tp),
        },
    }


def main() -> None:
    print("=" * 80)
    print("      HTTPS ROBUSTNESS EXPERIMENT: DOMAIN-SEPARATED BENCHMARK")
    print("=" * 80)

    EXP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXP_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    target_data = DATA_PATH if DATA_PATH.exists() else RAW_DATA_PATH
    print(f"Loading dataset from: {target_data}")

    cols = ["URL", "label"]
    if target_data == DATA_PATH:
        cols.append("registered_domain")

    df = pd.read_csv(target_data, usecols=cols, low_memory=False)
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["URL"] != ""].copy()

    print(f"Loaded {len(df):,} valid records.")

    # Registered domain extraction with fallback
    print("Resolving registered domains...")
    df["hostname"] = df["URL"].map(get_hostname)
    if "registered_domain" not in df.columns:
        df["registered_domain"] = df["URL"].map(get_registered_domain)
    else:
        df["registered_domain"] = (
            df["registered_domain"]
            .fillna(df["hostname"])
            .astype(str)
            .str.strip()
            .str.lower()
        )
    df["registered_domain"] = (
        df["registered_domain"]
        .replace({"": None, "nan": None, "none": None})
        .fillna(df["hostname"])
        .astype(str)
    )
    df = df[(df["registered_domain"] != "") & (df["hostname"] != "")].copy()

    print(f"Unique registered domains: {df['registered_domain'].nunique():,}")

    # Feature extraction
    print("\nExtracting all 19 lexical features for full dataset...")
    raw_features = [extract_features(url) for url in df["URL"]]
    X_full = pd.DataFrame(raw_features, index=df.index)
    y = df["label"]
    feature_names_baseline = X_full.columns.tolist()

    assert "https_present" in feature_names_baseline, "https_present missing from features!"

    feature_names_exp = [f for f in feature_names_baseline if f != "https_present"]
    X_exp = X_full[feature_names_exp].copy()

    print(f"Baseline feature count    : {len(feature_names_baseline)} features")
    print(f"Experimental feature count: {len(feature_names_exp)} features (https_present excluded)")

    # Registered domain partition
    print("\nCreating domain-separated train/test split (80/20, seed=42)...")
    groups = df["registered_domain"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X_full, y, groups=groups))

    train_domains = set(groups.iloc[train_idx])
    test_domains = set(groups.iloc[test_idx])
    domain_overlap = train_domains & test_domains

    print(f"Train set: {len(train_idx):,} rows ({len(train_domains):,} domains)")
    print(f"Test set : {len(test_idx):,} rows ({len(test_domains):,} domains)")
    print(f"Domain overlap check: {len(domain_overlap)} overlapping domains")

    if len(domain_overlap) > 0:
        raise RuntimeError(f"FATAL: Domain leakage detected: {domain_overlap}")

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_urls = df["URL"].iloc[test_idx].reset_index(drop=True)
    test_https = X_full["https_present"].iloc[test_idx].reset_index(drop=True)

    print("\nTrain class distribution:")
    print(f"  Phishing (0)   : {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Legitimate (1) : {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    print("Test class distribution:")
    print(f"  Phishing (0)   : {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"  Legitimate (1) : {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.1f}%)")

    # 1. Train & evaluate Baseline Model
    print("\n" + "-" * 75)
    print("1. TRAINING BASELINE MODEL (19 Features with https_present)")
    print("-" * 75)
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    baseline_model.fit(X_full.iloc[train_idx], y_train)
    y_pred_base = baseline_model.predict(X_full.iloc[test_idx])
    base_metrics = compute_metrics(y_test, y_pred_base, "Baseline (with https_present)")

    # 2. Train & evaluate Experimental Model
    print("\n" + "-" * 75)
    print("2. TRAINING EXPERIMENTAL MODEL (18 Features without https_present)")
    print("-" * 75)
    exp_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    exp_model.fit(X_exp.iloc[train_idx], y_train)
    y_pred_exp = exp_model.predict(X_exp.iloc[test_idx])
    exp_metrics = compute_metrics(y_test, y_pred_exp, "Experimental (no https_present)")

    # Save experimental model separately
    exp_model_path = EXP_MODEL_DIR / "url_rf_no_https.joblib"
    joblib.dump(
        {
            "model": exp_model,
            "feature_names": feature_names_exp,
            "label_values": [0, 1],
            "description": "Experimental Random Forest trained on 18 lexical features (excluding https_present)",
        },
        exp_model_path,
    )
    print(f"\nExperimental model saved to: {exp_model_path}")

    # Deep dive: Error analysis by HTTPS scheme presence
    test_df_eval = pd.DataFrame({
        "url": test_urls,
        "is_https": test_https,
        "true_label": y_test.values,
        "pred_base": y_pred_base,
        "pred_exp": y_pred_exp,
    })

    # HTTPS phishing (actual 0, is_https == 1): How many were missed (pred == 1)?
    https_phish = test_df_eval[(test_df_eval["true_label"] == 0) & (test_df_eval["is_https"] == 1)]
    https_phish_missed_base = (https_phish["pred_base"] == 1).sum()
    https_phish_missed_exp = (https_phish["pred_exp"] == 1).sum()

    # HTTP legitimate (actual 1, is_https == 0): How many were falsely blocked (pred == 0)?
    http_legit = test_df_eval[(test_df_eval["true_label"] == 1) & (test_df_eval["is_https"] == 0)]
    http_legit_blocked_base = (http_legit["pred_base"] == 0).sum()
    http_legit_blocked_exp = (http_legit["pred_exp"] == 0).sum()

    error_analysis = {
        "https_phishing_total": int(len(https_phish)),
        "https_phishing_missed_by_baseline": int(https_phish_missed_base),
        "https_phishing_missed_by_experimental": int(https_phish_missed_exp),
        "http_legitimate_total": int(len(http_legit)),
        "http_legitimate_blocked_by_baseline": int(http_legit_blocked_base),
        "http_legitimate_blocked_by_experimental": int(http_legit_blocked_exp),
    }

    # Print comprehensive comparison
    print("\n" + "=" * 90)
    print("                 HEAD-TO-HEAD MODEL COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<32} {'Baseline (+HTTPS)':<26} {'Experimental (-HTTPS)':<26} {'Delta':<12}")
    print("-" * 90)

    acc_delta = exp_metrics["accuracy"] - base_metrics["accuracy"]
    macro_delta = exp_metrics["macro_f1"] - base_metrics["macro_f1"]
    p_prec_delta = exp_metrics["phishing"]["precision"] - base_metrics["phishing"]["precision"]
    p_rec_delta = exp_metrics["phishing"]["recall"] - base_metrics["phishing"]["recall"]
    p_f1_delta = exp_metrics["phishing"]["f1"] - base_metrics["phishing"]["f1"]
    l_prec_delta = exp_metrics["legitimate"]["precision"] - base_metrics["legitimate"]["precision"]
    l_rec_delta = exp_metrics["legitimate"]["recall"] - base_metrics["legitimate"]["recall"]
    l_f1_delta = exp_metrics["legitimate"]["f1"] - base_metrics["legitimate"]["f1"]

    print(f"{'Accuracy':<32} {base_metrics['accuracy']*100:>10.2f}%               {exp_metrics['accuracy']*100:>10.2f}%               {acc_delta*100:>+6.2f}%")
    print(f"{'Macro F1':<32} {base_metrics['macro_f1']*100:>10.2f}%               {exp_metrics['macro_f1']*100:>10.2f}%               {macro_delta*100:>+6.2f}%")
    print(f"{'Phishing Precision (0)':<32} {base_metrics['phishing']['precision']*100:>10.2f}%               {exp_metrics['phishing']['precision']*100:>10.2f}%               {p_prec_delta*100:>+6.2f}%")
    print(f"{'Phishing Recall (0)':<32} {base_metrics['phishing']['recall']*100:>10.2f}%               {exp_metrics['phishing']['recall']*100:>10.2f}%               {p_rec_delta*100:>+6.2f}%")
    print(f"{'Phishing F1-Score (0)':<32} {base_metrics['phishing']['f1']*100:>10.2f}%               {exp_metrics['phishing']['f1']*100:>10.2f}%               {p_f1_delta*100:>+6.2f}%")
    print(f"{'Legitimate Precision (1)':<32} {base_metrics['legitimate']['precision']*100:>10.2f}%               {exp_metrics['legitimate']['precision']*100:>10.2f}%               {l_prec_delta*100:>+6.2f}%")
    print(f"{'Legitimate Recall (1)':<32} {base_metrics['legitimate']['recall']*100:>10.2f}%               {exp_metrics['legitimate']['recall']*100:>10.2f}%               {l_rec_delta*100:>+6.2f}%")
    print(f"{'Legitimate F1-Score (1)':<32} {base_metrics['legitimate']['f1']*100:>10.2f}%               {exp_metrics['legitimate']['f1']*100:>10.2f}%               {l_f1_delta*100:>+6.2f}%")

    print("-" * 90)
    print("CONFUSION MATRICES & ERROR BREAKDOWN:")
    print(f"{'Metric':<32} {'Baseline (+HTTPS)':<26} {'Experimental (-HTTPS)':<26} {'Delta (Counts)':<12}")
    print("-" * 90)
    print(f"{'True Phishing Detected (0->0)':<32} {base_metrics['phishing']['detected']:>10,}               {exp_metrics['phishing']['detected']:>10,}               {exp_metrics['phishing']['detected'] - base_metrics['phishing']['detected']:>+6,}")
    print(f"{'Missed Phishing (0->1, FP)':<32} {base_metrics['phishing']['missed_false_positive']:>10,}               {exp_metrics['phishing']['missed_false_positive']:>10,}               {exp_metrics['phishing']['missed_false_positive'] - base_metrics['phishing']['missed_false_positive']:>+6,}")
    print(f"{'Blocked Legitimate (1->0, FN)':<32} {base_metrics['legitimate']['blocked_false_alarm']:>10,}               {exp_metrics['legitimate']['blocked_false_alarm']:>10,}               {exp_metrics['legitimate']['blocked_false_alarm'] - base_metrics['legitimate']['blocked_false_alarm']:>+6,}")
    print(f"{'True Legitimate Passed (1->1)':<32} {base_metrics['legitimate']['passed']:>10,}               {exp_metrics['legitimate']['passed']:>10,}               {exp_metrics['legitimate']['passed'] - base_metrics['legitimate']['passed']:>+6,}")

    print("\n" + "-" * 90)
    print("SCHEME-SPECIFIC FAILURE ANALYSIS:")
    print(f"Total HTTPS Phishing URLs in Test Set: {error_analysis['https_phishing_total']:,}")
    print(f"  - Missed by Baseline Model        : {error_analysis['https_phishing_missed_by_baseline']:,} ({error_analysis['https_phishing_missed_by_baseline']/max(error_analysis['https_phishing_total'],1)*100:.2f}%)")
    print(f"  - Missed by Experimental Model    : {error_analysis['https_phishing_missed_by_experimental']:,} ({error_analysis['https_phishing_missed_by_experimental']/max(error_analysis['https_phishing_total'],1)*100:.2f}%)")
    print(f"Total HTTP Legitimate URLs in Test Set: {error_analysis['http_legitimate_total']:,}")
    print(f"  - Falsely Blocked by Baseline     : {error_analysis['http_legitimate_blocked_by_baseline']:,} ({error_analysis['http_legitimate_blocked_by_baseline']/max(error_analysis['http_legitimate_total'],1)*100:.2f}%)")
    print(f"  - Falsely Blocked by Experimental : {error_analysis['http_legitimate_blocked_by_experimental']:,} ({error_analysis['http_legitimate_blocked_by_experimental']/max(error_analysis['http_legitimate_total'],1)*100:.2f}%)")

    # Feature Importance comparison
    base_imp = pd.Series(baseline_model.feature_importances_, index=feature_names_baseline).sort_values(ascending=False)
    exp_imp = pd.Series(exp_model.feature_importances_, index=feature_names_exp).sort_values(ascending=False)

    print("\n" + "=" * 90)
    print("FEATURE IMPORTANCE RE-DISTRIBUTION (Top 10)")
    print("=" * 90)
    print(f"{'Rank':<5} {'Baseline Feature':<25} {'Weight':<10} | {'Experimental Feature':<25} {'Weight':<10}")
    print("-" * 90)
    for i in range(10):
        b_name = base_imp.index[i]
        b_val = base_imp.iloc[i]
        e_name = exp_imp.index[i]
        e_val = exp_imp.iloc[i]
        print(f"{i+1:<5} {b_name:<25} {b_val*100:>6.2f}%   | {e_name:<25} {e_val*100:>6.2f}%")

    # Save artifacts
    results_payload = {
        "dataset_rows": len(df),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_domains": len(train_domains),
        "test_domains": len(test_domains),
        "domain_overlap": len(domain_overlap),
        "baseline_metrics": base_metrics,
        "experimental_metrics": exp_metrics,
        "error_analysis": error_analysis,
        "feature_importance_baseline": base_imp.to_dict(),
        "feature_importance_experimental": exp_imp.to_dict(),
    }

    results_json_path = EXP_ARTIFACTS_DIR / "experiment_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Save human-readable report
    report_txt_path = EXP_ARTIFACTS_DIR / "experiment_report.txt"
    report_lines = [
        "HTTPS ROBUSTNESS EXPERIMENT REPORT",
        "==================================",
        f"Evaluated records: {len(df):,}",
        f"Domain-separated test split: {len(test_idx):,} URLs ({len(test_domains):,} registered domains)",
        f"Domain overlap: {len(domain_overlap)}",
        "",
        "COMPARISON SUMMARY:",
        f"  Baseline Accuracy     : {base_metrics['accuracy']*100:.2f}%",
        f"  Experimental Accuracy : {exp_metrics['accuracy']*100:.2f}% (Delta: {acc_delta*100:+.2f}%)",
        f"  Baseline Macro F1     : {base_metrics['macro_f1']*100:.2f}%",
        f"  Experimental Macro F1 : {exp_metrics['macro_f1']*100:.2f}% (Delta: {macro_delta*100:+.2f}%)",
        "",
        f"  Phishing Recall (0)   : Baseline {base_metrics['phishing']['recall']*100:.2f}% -> Experimental {exp_metrics['phishing']['recall']*100:.2f}% ({p_rec_delta*100:+.2f}%)",
        f"  Phishing Prec   (0)   : Baseline {base_metrics['phishing']['precision']*100:.2f}% -> Experimental {exp_metrics['phishing']['precision']*100:.2f}% ({p_prec_delta*100:+.2f}%)",
        "",
        "ERROR COUNTS:",
        f"  Missed Phishing (FP)  : Baseline {base_metrics['phishing']['missed_false_positive']} -> Experimental {exp_metrics['phishing']['missed_false_positive']} ({exp_metrics['phishing']['missed_false_positive'] - base_metrics['phishing']['missed_false_positive']:+d})",
        f"  False Alarms (FN)     : Baseline {base_metrics['legitimate']['blocked_false_alarm']} -> Experimental {exp_metrics['legitimate']['blocked_false_alarm']} ({exp_metrics['legitimate']['blocked_false_alarm'] - base_metrics['legitimate']['blocked_false_alarm']:+d})",
        "",
        "SCHEME ANALYSIS:",
        f"  HTTPS Phishing Missed : Baseline {error_analysis['https_phishing_missed_by_baseline']} / {error_analysis['https_phishing_total']} -> Experimental {error_analysis['https_phishing_missed_by_experimental']} / {error_analysis['https_phishing_total']}",
        f"  HTTP Legit Blocked    : Baseline {error_analysis['http_legitimate_blocked_by_baseline']} / {error_analysis['http_legitimate_total']} -> Experimental {error_analysis['http_legitimate_blocked_by_experimental']} / {error_analysis['http_legitimate_total']}",
    ]
    report_txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETE")
    print(f"Results JSON  : {results_json_path}")
    print(f"Results Report: {report_txt_path}")
    print(f"Model Artifact: {exp_model_path}")
    print("Production models and datasets remain untouched.")
    print("=" * 90)


if __name__ == "__main__":
    main()
