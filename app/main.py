"""Lightweight Streamlit app for Vietnamese food review sentiment."""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import MODEL_PATH, load_model, predict_review


st.set_page_config(page_title="Đọc vị review món ăn", page_icon="🍜", layout="centered")
style = (ROOT / "asset" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
st.caption("MINI PROJECT  /  NLP TIẾNG VIỆT")
st.title("Đọc vị review món ăn")
st.write("Nhập một bình luận để xem mô hình xếp nó vào nhóm tích cực hay tiêu cực.")
st.divider()


@st.cache_resource(show_spinner="Thinking...")
def cached_model(model_modified_ns: int):
    # The file timestamp invalidates Streamlit's cache after retraining.
    return load_model()


if not MODEL_PATH.is_file():
    st.error("Chưa có model. Tải dữ liệu Kaggle vào `data/raw/`, rồi chạy `python -m model.main`.")
    st.stop()

example = st.selectbox(
    "Thử nhanh một câu mẫu",
    [
        "Tự nhập review",
        "Phở ngon, nước dùng đậm đà và nhân viên rất nhiệt tình.",
        "Đồ ăn nguội, phục vụ quá chậm và giá đắt.",
        "Món ăn ngon nhưng phải đợi khá lâu mới được phục vụ.",
    ],
)
initial_text = "" if example == "Tự nhập review" else example
with st.form("review_form"):
    review = st.text_area("Review", value=initial_text, height=130, max_chars=5_000)
    submitted = st.form_submit_button("Dự đoán", type="primary")

if submitted:
    if not review.strip():
        st.warning("Bạn hãy nhập một review trước nhé.")
    else:
        result = predict_review(cached_model(MODEL_PATH.stat().st_mtime_ns), review)
        st.divider()
        st.caption("KẾT QUẢ DỰ ĐOÁN")
        st.subheader(result["label"])
        st.metric("Xác suất kỹ thuật cho nhãn này", f"{result['probability']:.1%}")
        chart_data = pd.DataFrame(
            {
                "Nhãn": ["Tích cực", "Tiêu cực"],
                "Xác suất": [
                    result["probabilities"]["Tích cực"],
                    result["probabilities"]["Tiêu cực"],
                ],
            }
        )
        chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusEnd=6, height=26)
            .encode(
                x=alt.X(
                    "Xác suất:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", title=None),
                ),
                y=alt.Y("Nhãn:N", sort=["Tích cực", "Tiêu cực"], title=None),
                color=alt.Color(
                    "Nhãn:N",
                    scale=alt.Scale(
                        domain=["Tiêu cực", "Tích cực"],
                        range=["#B45338", "#3A806F"],
                    ),
                    legend=None,
                ),
                tooltip=["Nhãn:N", alt.Tooltip("Xác suất:Q", format=".1%")],
            )
            .properties(height=110)
        )
        st.altair_chart(chart, width="stretch", theme=None)
        st.caption(
            "Xác suất do mô hình ước lượng, chưa hiệu chỉnh; không phải mức độ cảm xúc. "
            "Ứng dụng chỉ phân loại hai nhãn, kể cả với review vừa khen vừa chê."
        )

report_path = ROOT / "model" / "training_report.json"
if report_path.is_file():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with st.expander("Mô hình được đánh giá như thế nào?"):
        st.write(f"Model: `{report['selected_model']}`")
        st.write(f"Số review duy nhất sau làm sạch: {report['cleaning']['clean_rows']:,}")
        st.write(f"Macro-F1 trên tập test: {report['test_scores']['macro_f1']:.3f}")
        if "curated_examples" in report:
            st.write("Đã thêm ví dụ tiếng lóng tự viết vào huấn luyện và kiểm tra trên 24 câu riêng.")
        st.write("Tập test được tách sau khi bỏ review trùng và không dùng để chọn mô hình.")
