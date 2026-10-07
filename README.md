# Vietnamese Food Review Sentiment

Ứng dụng phân loại cảm xúc của review ẩm thực tiếng Việt thành hai nhãn **Tích cực** và **Tiêu cực**. Dự án sử dụng TF-IDF, Logistic Regression và giao diện Streamlit; model đã được huấn luyện sẵn nên có thể chạy ngay sau khi cài thư viện.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows, kích hoạt môi trường ảo bằng:

```bash
.venv\Scripts\activate
```

## Chạy ứng dụng

```bash
python -m streamlit run app/main.py
```

Sau khi chạy, mở địa chỉ được hiển thị trong terminal (thường là `http://localhost:8501`) để sử dụng ứng dụng.

Ngoài ra, có thể dự đoán trực tiếp từ terminal:

```bash
python -m model.predict "Phở ngon, nước dùng đậm đà và phục vụ nhiệt tình."
```
