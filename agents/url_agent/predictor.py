
from pathlib import Path

import joblib
import pandas as pd

try:
    from agents.url_agent.feature_extractor import extract_features
except ModuleNotFoundError:
    from feature_extractor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "url_random_forest.joblib"

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1


def predict_url(url: str, model, feature_names: list) -> dict:
    """Run Random Forest prediction on a single URL."""
    features = pd.DataFrame([extract_features(url)])
    features = features.reindex(columns=feature_names)

    prediction = model.predict(features)[0]
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)[0]

    phishing_index = classes.index(PHISHING_LABEL)
    phishing_score = float(probabilities[phishing_index]) * 100

    if prediction == PHISHING_LABEL:
        prediction_text = "Potentially Phishing"
    elif prediction == LEGITIMATE_LABEL:
        prediction_text = "Likely Legitimate"
    else:
        prediction_text = f"Unknown model label: {prediction}"

    return {
        "url": url,
        "prediction": prediction_text,
        "predicted_label": int(prediction),
        "phishing_model_score_percent": phishing_score,
    }


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Train the model first:")
        print("python agents/url_agent/train_model.py")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    print("=" * 45)
    print("       RANDOM FOREST URL DETECTOR")
    print("=" * 45)
    print("Label 0 = Phishing | Label 1 = Legitimate")
    print("Enter 'quit' to exit.\n")

    while True:
        url = input("Enter URL: ").strip()

        if url.lower() in {"quit", "exit", "q"}:
            print("Exiting detector.")
            break

        if not url:
            print("Please enter a URL.\n")
            continue

        try:
            res = predict_url(url, model, feature_names)
            print(f"\nURL: {res['url']}")
            print(f"Prediction: {res['prediction']}")
            print(f"Model phishing score: {res['phishing_model_score_percent']:.2f}%")
            print(
                "Note: This score is not necessarily a calibrated "
                "real-world probability."
            )
            print("This is a URL-pattern check, not a guarantee of safety.\n")

        except Exception as exc:
            print(f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
