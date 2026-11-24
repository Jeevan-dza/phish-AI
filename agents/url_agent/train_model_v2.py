"""
PhishGuard AI – URL Model Training Pipeline  (v2)
===================================================
Trains an improved Random Forest on a balanced dataset that includes:
  - Phishing URLs from PhiUSIIL (label=0)
  - Legitimate homepages + synthetic deep-path URLs built from top-1m.csv (label=1)

Purpose of synthetic deep-path URLs:
  The PhiUSIIL legitimate set contains ONLY homepage-style URLs (zero path
  depth, no query strings).  This creates a severe train/serve distribution
  mismatch: at inference time the model receives deep URLs like
  accounts.google.com/signin or amazon.com/dp/B09X7?ref=…, which it has
  never seen as legitimate and therefore classifies as phishing.

  The synthetic set generates common legitimate URL patterns from verified
  top-ranked domains so the model learns that having a path, a query string,
  or a 'login' subdomain does NOT imply phishing by itself.

Steps:
  1. Load PhiUSIIL phishing + legitimate homepages.
  2. Remove duplicates and URL conflicts.
  3. Build synthetic deep legitimate URLs from top-1m domains.
  4. Combine → balanced dataset.
  5. Domain-separated train/test split (GroupShuffleSplit on registered domain)
     to prevent data leakage.
  6. Extract v2 features.
  7. Train RandomForest(n_estimators=300, max_depth=20, class_weight='balanced').
  8. Evaluate: accuracy, precision, recall, F1, confusion matrix, false-positive rate.
  9. Save model bundle to models/url_rf_v2.joblib.
 10. Back up old model to models/url_random_forest_v1_backup.joblib.
 11. Copy new model to models/url_random_forest.joblib (live model slot).

Usage:
    .venv\\Scripts\\python.exe agents/url_agent/train_model_v2.py

Estimated runtime: 3–8 minutes depending on hardware.
"""

from __future__ import annotations

import json
import shutil
import sys
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
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
import tldextract

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PHIUSIIL_PATH = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
TOP1M_PATH = PROJECT_ROOT / "data" / "top-1m.csv"
VERIFIED_DEEP_PATH = PROJECT_ROOT / "data" / "verified_legitimate" / "verified_legitimate_deep_urls.csv"
MODEL_V2_PATH = PROJECT_ROOT / "models" / "url_rf_v2.joblib"
LIVE_MODEL_PATH = PROJECT_ROOT / "models" / "url_random_forest.joblib"
BACKUP_PATH = PROJECT_ROOT / "models" / "url_random_forest_v1_backup.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "url_rf_v2_metrics.json"

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

_TLDEXT = tldextract.TLDExtract(suffix_list_urls=[], fallback_to_snapshot=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_registered_domain(url: str) -> str:
    val = str(url).strip()
    norm = val if "://" in val else "http://" + val
    try:
        host = (urlsplit(norm).hostname or "").lower()
    except Exception:
        host = ""
    if not host:
        return ""
    ext = _TLDEXT(host)
    rd = ext.top_domain_under_public_suffix or ext.registered_domain
    return (rd or host).lower()


def compute_metrics(y_true, y_pred, name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false-positive rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # false-negative rate

    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "phishing": {
            "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "true_positives": int(tn),
            "false_negatives_missed": int(fp),
        },
        "legitimate": {
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "true_positives": int(tp),
            "false_positives_blocked": int(fn),
        },
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "TN_phishing_correct": int(tn),
            "FP_phishing_missed_as_legit": int(fp),
            "FN_legit_blocked_as_phishing": int(fn),
            "TP_legit_correct": int(tp),
        },
    }


# ---------------------------------------------------------------------------
# Synthetic legitimate deep-path URL generator
# ---------------------------------------------------------------------------

PATH_CATEGORIES = [
    "personal", "business", "enterprise", "banking", "checking", "savings",
    "credit-cards", "investments", "loans", "mortgage", "insurance", "finance",
    "questions", "answers", "articles", "posts", "topics", "help", "support",
    "docs", "documentation", "developer", "developers", "api", "guides",
    "products", "items", "categories", "cart", "checkout", "orders", "shop",
    "blog", "news", "press", "events", "about", "contact-us", "careers",
    "privacy-policy", "terms-of-service", "security", "download", "en-us",
    "common", "oauth2", "v2.0", "v1", "authorize", "token", "auth", "login",
    "signin", "account", "accounts", "user", "users", "profile", "settings",
    "dashboard", "portal", "my-account", "overview", "details", "view"
]

SLUGS_AND_IDS = [
    "checking-and-savings", "how-to-use-python", "getting-started",
    "api-reference", "frequently-asked-questions", "savings",
    "12345", "987654", "2024", "2023", "HT213604", "B09X7MVJNH",
    "dp", "item-94821", "id-01928", "page-1", "user-profile"
]

LEGIT_SUBDOMAINS = [
    "",
    "www.",
    "accounts.",
    "account.",
    "mail.",
    "login.",
    "support.",
    "help.",
    "docs.",
    "developer.",
    "api.",
    "my.",
    "secure.",
    "app.",
    "portal.",
]

QUERY_TEMPLATES = [
    "",
    "",
    "",
    "?ref=nav",
    "?page=2&sort=desc",
    "?q=search+term",
    "?id=102&tab=overview",
    "?redirect_uri=https&client_id=web",
    "?lang=en-us",
    "?ref=ppx_yo2ov",
]


def generate_synthetic_legit_urls(top1m_path: Path, n: int = 75_000) -> pd.DataFrame:
    """
    Build synthetic legitimate URLs from top-ranked domains with realistic
    path depth, subdomains, hyphenated slugs, numeric IDs, and query parameters.
    """
    print(f"  Loading top-1m from {top1m_path} …")
    top1m = pd.read_csv(top1m_path, header=None, names=["rank", "domain"])
    # Use top 75,000 domains
    top1m = top1m[top1m["rank"] <= 75_000].copy()
    top1m["domain"] = top1m["domain"].str.strip().str.lower()

    rng = np.random.default_rng(42)
    sampled_domains = top1m["domain"].sample(n=n, replace=True, random_state=42).values
    subdomains = rng.choice(LEGIT_SUBDOMAINS, size=n)
    schemes = rng.choice(["https", "https", "https", "https", "http"], size=n)  # ~80% https, 20% http
    queries = rng.choice(QUERY_TEMPLATES, size=n)

    urls = []
    pool = PATH_CATEGORIES + SLUGS_AND_IDS

    for i in range(n):
        dom = sampled_domains[i]
        sub = subdomains[i]
        scheme = schemes[i]
        q = queries[i]
        # Depth 0 to 4
        depth = rng.integers(0, 5)
        if depth == 0:
            path = ""
        else:
            segs = rng.choice(pool, size=depth)
            path = "/" + "/".join(segs)
        urls.append(f"{scheme}://{sub}{dom}{path}{q}")

    return pd.DataFrame({"URL": urls, "label": LEGITIMATE_LABEL})


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  PhishGuard AI – URL Model Training Pipeline v2")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load PhiUSIIL dataset
    # ------------------------------------------------------------------
    print("\n[1/9] Loading PhiUSIIL dataset …")
    df_raw = pd.read_csv(PHIUSIIL_PATH, low_memory=False)
    print(f"  Loaded {len(df_raw):,} rows. Columns include: {list(df_raw.columns[:6])}")

    df_raw = df_raw[["URL", "label"]].copy()
    df_raw["URL"] = df_raw["URL"].astype(str).str.strip()
    df_raw["label"] = df_raw["label"].astype(int)

    # ------------------------------------------------------------------
    # 2. Remove duplicates from PhiUSIIL
    # ------------------------------------------------------------------
    print("\n[2/9] Deduplication and quality checks …")
    before = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset="URL")
    after_dedup = len(df_raw)
    print(f"  Removed {before - after_dedup} duplicate URLs ({before:,} → {after_dedup:,})")

    # Remove empty/malformed URLs
    df_raw = df_raw[df_raw["URL"].str.contains(".", na=False)]
    print(f"  After malformed removal: {len(df_raw):,}")

    phiusiil_phish = df_raw[df_raw["label"] == 0].copy()
    phiusiil_legit = df_raw[df_raw["label"] == 1].copy()
    print(f"  PhiUSIIL phishing: {len(phiusiil_phish):,}  legitimate: {len(phiusiil_legit):,}")

    # ------------------------------------------------------------------
    # 3. Generate synthetic deep legitimate URLs
    # ------------------------------------------------------------------
    print("\n[3/9] Generating synthetic deep legitimate URLs from top-1m …")
    df_synth = generate_synthetic_legit_urls(TOP1M_PATH, n=80_000)
    print(f"  Generated {len(df_synth):,} synthetic legitimate URLs")

    # Remove any synthetic URL whose domain overlaps with phishing domains
    # to avoid confusion in the training set
    phish_domains = set(phiusiil_phish["URL"].apply(get_registered_domain))
    df_synth["reg_domain"] = df_synth["URL"].apply(get_registered_domain)
    before_filter = len(df_synth)
    df_synth = df_synth[~df_synth["reg_domain"].isin(phish_domains)]
    print(f"  Removed {before_filter - len(df_synth):,} synthetic URLs overlapping phish domains")
    print(f"  Clean synthetic legit: {len(df_synth):,}")

    # ------------------------------------------------------------------
    # 4. Combine dataset
    # ------------------------------------------------------------------
    print("\n[4/9] Combining datasets …")
    # Subsample PhiUSIIL phishing to 100k
    phish_sample = phiusiil_phish.sample(
        n=min(100_000, len(phiusiil_phish)), random_state=42
    )[["URL", "label"]]

    # PhiUSIIL legit (homepage-only) – cap at 40k to avoid dominating
    legit_homepage = phiusiil_legit.sample(
        n=min(40_000, len(phiusiil_legit)), random_state=42
    )[["URL", "label"]]

    # Synthetic deep legit – use full set
    legit_deep = df_synth[["URL", "label"]].copy()

    datasets_to_combine = [phish_sample, legit_homepage, legit_deep]

    # Include verified legitimate deep URLs if present
    if VERIFIED_DEEP_PATH.exists():
        try:
            df_vdeep = pd.read_csv(VERIFIED_DEEP_PATH)
            col = "raw_url" if "raw_url" in df_vdeep.columns else "URL"
            df_vdeep = df_vdeep[[col]].rename(columns={col: "URL"})
            df_vdeep["label"] = LEGITIMATE_LABEL
            datasets_to_combine.append(df_vdeep)
            print(f"  Added {len(df_vdeep):,} verified legitimate deep URLs from {VERIFIED_DEEP_PATH.name}")
        except Exception as e:
            print(f"  Could not load verified deep URLs: {e}")

    df_combined = pd.concat(datasets_to_combine, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset="URL").reset_index(drop=True)

    label_counts = df_combined["label"].value_counts()
    print(f"  Combined dataset: {len(df_combined):,} rows")
    print(f"  Label 0 (phishing):  {label_counts.get(0, 0):,}")
    print(f"  Label 1 (legitimate):{label_counts.get(1, 0):,}")

    # ------------------------------------------------------------------
    # 5. Domain-separated train/test split (prevent leakage)
    # ------------------------------------------------------------------
    print("\n[5/9] Domain-separated train/test split (80/20, seed 42) …")
    df_combined["reg_domain"] = df_combined["URL"].apply(get_registered_domain)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        gss.split(df_combined, df_combined["label"], groups=df_combined["reg_domain"])
    )
    train_df = df_combined.iloc[train_idx].copy()
    test_df = df_combined.iloc[test_idx].copy()
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ------------------------------------------------------------------
    # 6. Feature extraction (v2)
    # ------------------------------------------------------------------
    print("\n[6/9] Extracting v2 features …")
    from agents.url_agent.feature_extractor import extract_features

    def safe_extract(url):
        try:
            return extract_features(url)
        except Exception:
            return None

    print("  Extracting training features …")
    train_feats = train_df["URL"].apply(safe_extract)
    valid_train = train_feats.notna()
    X_train = pd.DataFrame(list(train_feats[valid_train]))
    y_train = train_df["label"][valid_train].values

    print("  Extracting test features …")
    test_feats = test_df["URL"].apply(safe_extract)
    valid_test = test_feats.notna()
    X_test = pd.DataFrame(list(test_feats[valid_test]))
    y_test = test_df["label"][valid_test].values

    feature_names = list(X_train.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 7. Train Random Forest v2
    # ------------------------------------------------------------------
    print("\n[7/9] Training Random Forest v2 …")
    print("  Parameters: n_estimators=300, max_depth=16, class_weight='balanced', "
          "min_samples_leaf=8, random_state=42")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,          # prevents overfitting to deep training URLs
        min_samples_leaf=8,    # requires at least 8 samples per leaf
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 8. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n[8/9] Evaluating on test set …")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, "RandomForest_v2")

    print(f"\n  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Phishing precision:  {metrics['phishing']['precision']:.4f}")
    print(f"  Phishing recall:     {metrics['phishing']['recall']:.4f}")
    print(f"  Phishing F1:         {metrics['phishing']['f1']:.4f}")
    print(f"  Legitimate precision:{metrics['legitimate']['precision']:.4f}")
    print(f"  Legitimate recall:   {metrics['legitimate']['recall']:.4f}")
    print(f"  Legitimate F1:       {metrics['legitimate']['f1']:.4f}")
    print(f"  False-Positive Rate: {metrics['false_positive_rate']:.4f}  "
          f"(legit blocked as phishing)")
    print(f"  False-Negative Rate: {metrics['false_negative_rate']:.4f}  "
          f"(phishing missed)")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred=Phish  Pred=Legit")
    print(f"  Act=Phish     {cm['TN_phishing_correct']:7,}  {cm['FP_phishing_missed_as_legit']:7,}")
    print(f"  Act=Legit     {cm['FN_legit_blocked_as_phishing']:7,}  {cm['TP_legit_correct']:7,}")

    print("\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Phishing(0)", "Legitimate(1)"]))

    # Feature importance
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Top feature importances:")
    for feat, imp in importances[:10]:
        print(f"    {feat:<30} {imp:.4f}")

    # Spot-check on known FP cases (run WITHIN the script – no network calls)
    print("\n  Spot-check on known false-positive URLs:")
    spot_urls = [
        ("https://accounts.google.com/signin", "legitimate"),
        ("https://www.amazon.com/dp/B09X7MVJNH?ref=ppx_yo2ov", "legitimate"),
        ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize", "legitimate"),
        ("https://stackoverflow.com/questions/12345/how-to-use-python", "legitimate"),
        ("https://www.chase.com/personal/banking", "legitimate"),
        ("https://support.apple.com/en-us/HT213604", "legitimate"),
        ("https://www.wellsfargo.com/checking-and-savings/savings/", "legitimate"),
        ("http://paypal-secure-login.verify-account.com/confirm", "phishing"),
        ("http://amazon-update-billing.tk/account/login", "phishing"),
        ("http://secure-bankofamerica.login-verify.ml/index.php", "phishing"),
    ]
    classes = list(model.classes_)
    phish_idx = classes.index(0)
    correct = 0
    for url, expected in spot_urls:
        try:
            from agents.url_agent.feature_extractor import extract_features
            feats = pd.DataFrame([extract_features(url)])
            feats = feats.reindex(columns=feature_names, fill_value=0)
            pred = model.predict(feats)[0]
            prob = model.predict_proba(feats)[0][phish_idx] * 100
            pred_label = "phishing" if pred == 0 else "legitimate"
            ok = pred_label == expected
            correct += int(ok)
            flag = "✓" if ok else "✗ WRONG"
            print(f"    {flag}  {url[:55]:<55} | exp={expected:10} | pred={pred_label:10} | {prob:5.1f}%")
        except Exception as e:
            print(f"    ERROR on {url}: {e}")

    print(f"\n  Spot-check: {correct}/{len(spot_urls)} correct")

    # ------------------------------------------------------------------
    # 9. Save model and metrics
    # ------------------------------------------------------------------
    print("\n[9/9] Saving model …")

    # Back up old live model
    if LIVE_MODEL_PATH.exists():
        shutil.copy2(LIVE_MODEL_PATH, BACKUP_PATH)
        print(f"  Backed up old model → {BACKUP_PATH.name}")

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "version": "v2",
        "training_notes": (
            "Trained on PhiUSIIL (phishing + legit homepages) "
            "plus synthetic deep legitimate URLs from top-1m.csv. "
            "Domain-separated 80/20 split. Features: v2 (22 features). "
            "max_depth=20, min_samples_leaf=5 to reduce overfitting."
        ),
    }
    joblib.dump(bundle, MODEL_V2_PATH)
    shutil.copy2(MODEL_V2_PATH, LIVE_MODEL_PATH)
    print(f"  Saved v2 model → {MODEL_V2_PATH.name}")
    print(f"  Updated live model → {LIVE_MODEL_PATH.name}")

    metrics["training_notes"] = bundle["training_notes"]
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics → {METRICS_PATH.name}")

    print("\n" + "=" * 70)
    print("  Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
