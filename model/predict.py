"""Shared inference helper and a small command-line demo."""

import argparse
from pathlib import Path

import joblib

from model.text import normalize_text


MODEL_PATH = Path(__file__).resolve().parent / "sentiment_pipeline.joblib"
LABELS = {0: "Tiêu cực", 1: "Tích cực"}


def load_model(path: Path = MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có model tại {path}. Hãy chạy: python -m model.main")
    return joblib.load(path)


def predict_review(model, review: str) -> dict:
    clean_review = normalize_text(review)
    if not clean_review:
        raise ValueError("Vui lòng nhập một review có nội dung.")
    prediction = int(model.predict([clean_review])[0])
    probabilities = model.predict_proba([clean_review])[0]
    probability_by_label = {
        LABELS[int(label)]: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    return {
        "label": LABELS[prediction],
        "probability": probability_by_label[LABELS[prediction]],
        "probabilities": probability_by_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc của một review tiếng Việt")
    parser.add_argument("review", help="Review cần phân loại")
    args = parser.parse_args()
    result = predict_review(load_model(), args.review)
    print(f"{result['label']} ({result['probability']:.1%})")


if __name__ == "__main__":
    main()
