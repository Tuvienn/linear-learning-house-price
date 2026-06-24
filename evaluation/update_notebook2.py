import json

filepath = "/Users/vientu/AI2026/Linear Learning/evaluation/linear_learning.ipynb"
with open(filepath, 'r') as f:
    nb = json.load(f)

markdown_updates = {
    0: """# House Price Prediction using Linear Regression

## 1. Data Collection & Understanding
### Define Problem
Mục tiêu của dự án này là xây dựng một mô hình **Linear Regression** cơ bản để dự đoán giá nhà dựa trên các đặc trưng (features) của ngôi nhà.
Đây là một bài toán **Hồi quy (Regression)** điển hình. Quá trình này được xây dựng bám sát theo 6 bước Machine Learning Pipeline.

**Biến mục tiêu (Target variable):** `SalePrice` - Giá bán thực tế của ngôi nhà.""",

    1: """### E2E Processing Flow
Dưới đây là luồng xử lý tổng thể (End-to-End Pipeline) bám sát 6 bước:
1. **Data Collection & Understanding**: Đọc dữ liệu thô từ `train.csv`, hiểu các cột và xác định biến mục tiêu `SalePrice`.
2. **Data Preproc
essing**: Làm sạch dữ liệu (missing values, duplicates, outliers), EDA và Feature Engineering (One-Hot Encoding).
3. **Data Splitting**: Chia tập huấn luyện (training set) và kiểm tra (test set) để đánh giá khách quan trên dữ liệu mới.
4. **Model Selection & Training**: Lựa chọn Linear Regression làm baseline model và huấn luyện.
5. **Evaluation & Tuning**: Đánh giá hiệu suất bằng MAE, MSE, RMSE, R² và trực quan hóa Actual vs Predicted, Residual Plot.
6. **Deployment & Monitoring**: Dự đoán mẫu thử, so sánh thực tế và định hướng theo dõi, cải thiện mô hình.""",

    14: """## 3. Data Splitting
Sử dụng `train_test_split` với `test_size=0.2` và `random_state=42`.
- **Training set (Tập huấn luyện)**: Được dùng để mô hình học các quy luật và mối quan hệ giữa các đặc trưng (features) và `SalePrice`.
- **Test set (Tập kiểm tra)**: Đóng vai trò là dữ liệu mới (unseen data) để đánh giá khả năng dự đoán thực tế của mô hình sau khi đã huấn luyện.""",

    16: """## 4. Model Selection & Training
Khởi tạo và huấn luyện (fit) mô hình **Linear Regression** trên tập `X_train` và `y_train`.

**Tại sao chọn Linear Regression làm baseline model?**
- Đơn giản, tốc độ huấn luyện rất nhanh.
- Dễ giải thích (interpretable) qua các hệ số (weights).
- Là thước đo cơ sở để so sánh với bất kỳ mô hình phức tạp nào sau này.""",

    18: """## 5. Evaluation & Tuning
Dự đoán trên tập `X_test` và tính toán các độ đo: **MAE**, **MSE**, **RMSE**, **R² Score**.
Đồng thời trực quan hóa **Actual vs Predicted plot** và **Residual Plot**.

*Lưu ý: Các kỹ thuật như **Ridge Regression** hoặc **Lasso Regression** chỉ được xem như định hướng cải thiện (future improvements) để chống Overfitting, không áp dụng làm mô hình chính trong pipeline này.*""",

    20: """## 6. Deployment & Monitoring
Hiển thị một vài kết quả dự đoán mẫu (Sample prediction) và so sánh trực tiếp với giá trị thực tế.

- **Ứng dụng**: Mô hình có thể được triển khai như một công cụ ước lượng giá bất động sản nhanh chóng dựa trên thông tin người dùng nhập vào.
- **Monitoring**: Trong thực tế, chúng ta cần theo dõi các sai số dự đoán (prediction errors) trên dữ liệu mới liên tục và cập nhật/huấn luyện lại mô hình trong tương lai.""",

    22: """### Conclusion
**Nhận xét về Baseline Model (Linear Regression):**
1. **R² Score** đạt khoảng **0.8805** (88.05%), chứng tỏ các biến đặc trưng giải thích được phần lớn sự biến động của giá nhà.
2. **RMSE** thực tế đạt khoảng **25.690 USD**. Việc này cho thấy vẫn còn độ lệch tương đối ở một số dự đoán (đặc biệt với các nhà giá trị cao).
3. Nhìn vào **Residual Plot**, sai số tăng dần theo giá trị thực tế của căn nhà, cho thấy Linear Regression bị ảnh hưởng nhiều bởi các ngoại lệ và không hoàn toàn bắt được các mối quan hệ phi tuyến tính.

**Hướng cải thiện (Future Work):**
- Xử lý thêm các biến lệch (Skewed variables) bằng phép biến đổi logarit (log-transform).
- Thử nghiệm Ridge/Lasso Regression như đã đề cập ở trên để kiểm soát độ phức tạp của mô hình."""
}

markdown_indices = [idx for idx, cell in enumerate(nb.get("cells", [])) if cell["cell_type"] == "markdown"]

for idx, cell in enumerate(nb.get("cells", [])):
    if idx in markdown_updates:
        # We split the new text by lines and append \n properly for Jupyter
        lines = markdown_updates[idx].split('\n')
        new_source = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]
        cell["source"] = new_source

with open(filepath, 'w') as f:
    json.dump(nb, f, indent=1)
    
print("Successfully updated the notebook.")
