"""Compare small text classifiers, select on validation, evaluate once on test."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from model.data import load_reviews
from model.text import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "vsa_food_rv_train.csv"
ARTIFACT_DIR = ROOT / "model"
CURATED_DIR = ROOT / "data" / "curated"
SEED = 42
FOCUS_REVIEW = "quán này ăn quá dở, mình không muốn quay lại nữa"


def make_candidates() -> dict[str, Pipeline]:
    word_features = dict(
        preprocessor=normalize_text,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=80_000,
        sublinear_tf=True,
    )
    return {
        "word_logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**word_features)),
                ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
            ]
        ),
        "word_naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**word_features)),
                ("classifier", MultinomialNB(alpha=1.0)),
            ]
        ),
        "char_logistic_regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        preprocessor=normalize_text,
                        lowercase=False,
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
            ]
        ),
    }


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "negative_f1": round(float(f1_score(y_true, y_pred, pos_label=0)), 4),
        "positive_f1": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
    }


def load_curated_sets(kaggle_texts: set[str]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read human-written examples and reject any train/evaluation overlap."""
    frames = {}
    audits = {}
    seen = set(kaggle_texts)
    for split in ("train", "validation", "test"):
        path = CURATED_DIR / f"slang_{split}.csv"
        frame, audit = load_reviews(path)
        if audit["clean_rows"] != audit["raw_rows"]:
            raise ValueError(f"{path}: dữ liệu tự viết có dòng lỗi hoặc trùng.")
        overlap = seen.intersection(frame["text"])
        if overlap:
            raise ValueError(f"{path}: có {len(overlap)} câu trùng với Kaggle hoặc split khác.")
        seen.update(frame["text"])
        frames[split] = frame
        audits[split] = {
            "rows": len(frame),
            "negative": int(frame["label"].eq(0).sum()),
            "positive": int(frame["label"].eq(1).sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return frames, audits


def fit_model(model: Pipeline, x_main, y_main, curated: pd.DataFrame, weight: int) -> Pipeline:
    """Give each extra example a declared weight, without duplicating its text."""
    if weight == 0:
        return model.fit(x_main, y_main)
    x_all = list(x_main) + curated["text"].tolist()
    y_all = np.concatenate([y_main, curated["label"].to_numpy()])
    weights = np.concatenate([np.ones(len(x_main)), np.full(len(curated), weight)])
    return model.fit(x_all, y_all, classifier__sample_weight=weights)


def negative_probability(model: Pipeline, text: str) -> float:
    class_index = list(model.classes_).index(0)
    return round(float(model.predict_proba([text])[0][class_index]), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện bộ phân loại review đồ ăn tiếng Việt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    frame, audit = load_reviews(args.data)
    curated, curated_audit = load_curated_sets(set(frame["text"]))
    x = frame["text"].tolist()
    y = frame["label"].to_numpy()

    # The held-out test set stays untouched until model choice is complete.
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev
    )

    validation = {}
    for name, model in make_candidates().items():
        print(f"Đang đánh giá: {name}", flush=True)
        model.fit(x_train, y_train)
        validation[name] = scores(y_val, model.predict(x_val))
        print(f"  macro-F1 validation: {validation[name]['macro_f1']:.4f}", flush=True)

    # Negative F1 breaks ties because the minority class is the harder class.
    best_name = max(
        validation,
        key=lambda name: (validation[name]["macro_f1"], validation[name]["negative_f1"]),
    )

    # Select the augmentation amount using validation only. The curated test
    # set, including the focus review, is not part of model selection.
    augmentation_validation = {}
    slang_val_x = curated["validation"]["text"].tolist()
    slang_val_y = curated["validation"]["label"].to_numpy()
    for weight in (0, 1, 4):
        model = fit_model(make_candidates()[best_name], x_train, y_train, curated["train"], weight)
        augmentation_validation[str(weight)] = {
            "kaggle_validation": scores(y_val, model.predict(x_val)),
            "slang_validation": scores(slang_val_y, model.predict(slang_val_x)),
        }

    baseline_val = augmentation_validation["0"]
    eligible = [
        weight for weight in (1, 4)
        if augmentation_validation[str(weight)]["kaggle_validation"]["macro_f1"]
        >= baseline_val["kaggle_validation"]["macro_f1"] - 0.005
        and augmentation_validation[str(weight)]["slang_validation"]["macro_f1"]
        > baseline_val["slang_validation"]["macro_f1"]
    ]
    selected_weight = max(
        eligible,
        key=lambda weight: (
            augmentation_validation[str(weight)]["slang_validation"]["macro_f1"],
            augmentation_validation[str(weight)]["kaggle_validation"]["macro_f1"],
            -weight,
        ),
        default=0,
    )

    baseline_model = fit_model(make_candidates()[best_name], x_dev, y_dev, curated["train"], 0)
    slang_test_x = curated["test"]["text"].tolist()
    slang_test_y = curated["test"]["label"].to_numpy()
    baseline_test_scores = scores(y_test, baseline_model.predict(x_test))
    baseline_slang_test_scores = scores(slang_test_y, baseline_model.predict(slang_test_x))

    if selected_weight:
        final_curated = pd.concat([curated["train"], curated["validation"]], ignore_index=True)
        best_model = fit_model(
            make_candidates()[best_name], x_dev, y_dev, final_curated, selected_weight
        )
    else:
        best_model = baseline_model
    test_predictions = best_model.predict(x_test)
    test_scores = scores(y_test, test_predictions)
    slang_test_predictions = best_model.predict(slang_test_x)
    slang_test_scores = scores(slang_test_y, slang_test_predictions)

    report = {
        "dataset_url": "https://www.kaggle.com/datasets/guutran/vietnamese-sentiment-analysis-food-reviews",
        "source_file_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": SEED,
        "cleaning": audit,
        "split": {
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
            "method": "stratified random 64/16/20 after normalized-text deduplication",
        },
        "validation_scores": validation,
        "selection_metric": "validation macro-F1; negative-class F1 as tie breaker",
        "selected_model": (
            best_name if not selected_weight else f"{best_name} + slang examples (weight {selected_weight})"
        ),
        "curated_examples": {
            "provenance": "Manually written examples for this project; not from Kaggle",
            "audit": curated_audit,
            "augmentation_validation": augmentation_validation,
            "selection_rule": (
                "Choose the weight with highest slang validation macro-F1 among models "
                "whose Kaggle validation macro-F1 falls by no more than 0.005; "
                "otherwise retain the baseline."
            ),
            "selected_weight": selected_weight,
            "baseline_slang_test_scores": baseline_slang_test_scores,
            "selected_slang_test_scores": slang_test_scores,
            "slang_test_confusion_matrix": confusion_matrix(
                slang_test_y, slang_test_predictions, labels=[0, 1]
            ).tolist(),
            "focus_case": {
                "text": FOCUS_REVIEW,
                "true_label": "negative",
                "baseline_negative_probability": negative_probability(baseline_model, FOCUS_REVIEW),
                "selected_negative_probability": negative_probability(best_model, FOCUS_REVIEW),
            },
        },
        "baseline_test_scores": baseline_test_scores,
        "test_scores": test_scores,
        "test_confusion_matrix": {
            "label_order": ["negative", "positive"],
            "rows_are_true_labels": True,
            "values": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
        },
        "test_classification_report": classification_report(
            y_test, test_predictions, labels=[0, 1],
            target_names=["negative", "positive"], output_dict=True, zero_division=0
        ),
        "evaluation_note": (
            "The Kaggle test split is the same split used in the earlier project version. "
            "Augmentation was selected on validation, but these test results are an iterative comparison, "
            "not a new external benchmark. The 24 manually written slang test cases are small and synthetic."
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "sentiment_pipeline.joblib")
    (ARTIFACT_DIR / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nModel gốc: {best_name}")
    print(f"Trọng số bổ sung được chọn: {selected_weight}")
    print(f"Macro-F1 Kaggle test: {baseline_test_scores['macro_f1']:.4f} -> {test_scores['macro_f1']:.4f}")
    print(f"Macro-F1 tiếng lóng test: {baseline_slang_test_scores['macro_f1']:.4f} -> {slang_test_scores['macro_f1']:.4f}")
    print(
        "Câu kiểm tra, xác suất tiêu cực: "
        f"{negative_probability(baseline_model, FOCUS_REVIEW):.1%} -> "
        f"{negative_probability(best_model, FOCUS_REVIEW):.1%}"
    )
    print(f"Đã lưu model và báo cáo tại: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
