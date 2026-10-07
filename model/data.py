"""Load and audit the labeled Kaggle reviews."""

from pathlib import Path

import pandas as pd

from model.text import normalize_text


def load_reviews(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one consistently labeled row per normalized review.

    Exact/near duplicates would leak across random train/test splits. Reviews with
    conflicting labels are removed altogether instead of choosing an arbitrary label.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy tải CSV từ Kaggle vào data/raw/ trước.")

    frame = pd.read_csv(path)
    required = {"Comment", "Rating"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV cần các cột {sorted(required)}; hiện có {list(frame.columns)}")

    audit = {"raw_rows": len(frame)}
    frame = frame[["Comment", "Rating"]].copy()
    frame["Rating"] = pd.to_numeric(frame["Rating"], errors="coerce")
    frame = frame.dropna(subset=["Comment", "Rating"])
    frame = frame[frame["Rating"].isin([0, 1])].copy()
    frame["text"] = frame["Comment"].map(normalize_text)
    frame = frame[frame["text"].ne("")].copy()
    audit["valid_rows"] = len(frame)

    conflicting = frame.groupby("text")["Rating"].nunique()
    conflicting_texts = set(conflicting[conflicting.gt(1)].index)
    audit["conflicting_texts_removed"] = len(conflicting_texts)
    audit["conflicting_rows_removed"] = int(frame["text"].isin(conflicting_texts).sum())
    frame = frame[~frame["text"].isin(conflicting_texts)]
    audit["duplicate_rows_removed"] = int(frame.duplicated(subset="text").sum())
    frame = frame.drop_duplicates(subset="text", keep="first")

    result = frame[["text", "Rating"]].rename(columns={"Rating": "label"}).reset_index(drop=True)
    result["label"] = result["label"].astype(int)
    audit["clean_rows"] = len(result)
    audit["negative_rows"] = int(result["label"].eq(0).sum())
    audit["positive_rows"] = int(result["label"].eq(1).sum())
    if result["label"].nunique() != 2 or result["label"].value_counts().min() < 5:
        raise ValueError("Cần ít nhất 5 review cho mỗi nhãn 0 và 1 sau khi làm sạch.")
    return result, audit
