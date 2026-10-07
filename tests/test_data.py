import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from model.data import load_reviews
from model.predict import load_model
from model.text import normalize_text
from model.main import CURATED_DIR, FOCUS_REVIEW, negative_probability


class DataCleaningTests(unittest.TestCase):
    def test_normalize_text_preserves_vietnamese_accents_and_negation(self):
        self.assertEqual(normalize_text("  KHÔNG   ngon\n lắm  "), "không ngon lắm")

    def test_conflicting_and_duplicate_reviews_do_not_reach_training(self):
        rows = [
            ("Ngon", 1), (" ngon ", 1),
            ("Dở", 0), (" DỞ", 1),
            ("Tốt", 1), ("Hợp vị", 1), ("Tuyệt", 1), ("Rất ngon", 1),
            ("Tệ", 0), ("Không ổn", 0), ("Dở quá", 0), ("Rất tệ", 0), ("Đắt", 0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reviews.csv"
            pd.DataFrame(rows, columns=["Comment", "Rating"]).to_csv(path, index=False)
            reviews, audit = load_reviews(path)
        self.assertTrue(reviews["text"].is_unique)
        self.assertNotIn("dở", set(reviews["text"]))
        self.assertEqual(audit["conflicting_texts_removed"], 1)
        self.assertEqual(audit["duplicate_rows_removed"], 1)

    def test_focus_review_is_held_out_from_curated_training(self):
        train_reviews, _ = load_reviews(CURATED_DIR / "slang_train.csv")
        validation_reviews, _ = load_reviews(CURATED_DIR / "slang_validation.csv")
        test_reviews, _ = load_reviews(CURATED_DIR / "slang_test.csv")
        self.assertNotIn(FOCUS_REVIEW, set(train_reviews["text"]))
        self.assertNotIn(FOCUS_REVIEW, set(validation_reviews["text"]))
        self.assertIn(FOCUS_REVIEW, set(test_reviews["text"]))

    def test_saved_model_matches_report(self):
        model = load_model()
        report_path = Path(__file__).resolve().parents[1] / "model" / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIsInstance(model.named_steps["classifier"], LogisticRegression)
        focus_case = report["curated_examples"]["focus_case"]
        self.assertEqual(focus_case["text"], FOCUS_REVIEW)
        self.assertEqual(
            focus_case["selected_negative_probability"],
            negative_probability(model, FOCUS_REVIEW),
        )


if __name__ == "__main__":
    unittest.main()
