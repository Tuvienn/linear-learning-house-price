# 🏡 AI House Price Predictor (Dự đoán Giá Nhà AI)

Dự án này là một **Hệ thống định giá Bất Động Sản tự động** (ML Pipeline) hoàn chỉnh (End-to-End). Nó sử dụng Hồi quy tuyến tính (Linear Regression) kết hợp Ridge Regularization để dự đoán giá nhà dựa trên bộ dữ liệu nổi tiếng **Ames Housing**.

Dự án bao gồm việc làm sạch dữ liệu, xây dựng Pipeline học máy, và cuối cùng là triển khai bằng một Giao diện Web (Streamlit Premium Dashboard).

---

## 🚀 Các Tính Năng Nổi Bật (Features)
- **Data Preprocessing Pipeline:** Tự động xử lý Missing Values (Giá trị thiếu) bằng `SimpleImputer` và chuẩn hoá dữ liệu `StandardScaler` / `OneHotEncoder`.
- **Model Tuning:** Giải quyết Overfitting bằng mô hình **Ridge Regression** cùng việc tìm kiếm tham số bằng `GridSearchCV`. Áp dụng `np.log1p` để chuẩn hoá phân phối giá trị mục tiêu (SalePrice).
- **Premium Web Dashboard:** Triển khai Model dưới dạng ứng dụng Web trực quan với giao diện thân thiện, sử dụng `Streamlit`.
- **Base House Concept:** Người dùng chỉ cần nhập 5 thông số quan trọng (Diện tích, Năm xây, Chất lượng,...), hệ thống sẽ tự động bù đắp hơn 70 tính năng còn lại bằng trung vị (median) của thị trường.

---

## 📂 Cấu Trúc Dự Án (Project Structure)
```
Linear Regression/
│
├── data/                       # Chứa dữ liệu thô (Raw Data)
│   ├── train.csv               # Dữ liệu train (Ames Housing)
│   └── data_description.txt    # Từ điển dữ liệu (Data Dictionary)
│
├── evaluation/                 # Khu vực thử nghiệm và huấn luyện
│   └── linear_regression.ipynb # Notebook chứa 6 bước xây dựng ML Pipeline
│
├── models/                     # Thư mục chứa mô hình đã được lưu (.pkl)
│   └── linear_regression_model.pkl # Model "Trí khôn" của hệ thống
│
├── flowchart/                  # Sơ đồ thiết kế hệ thống (.puml, .png)
├── app.py                      # Mã nguồn Frontend (Streamlit Dashboard)
├── working_rule.md             # Quy tắc làm việc & triển khai
└── README.md                   # Tài liệu hướng dẫn này
```

---

## 🛠️ Hướng dẫn Cài đặt & Chạy (Getting Started)

### Yêu cầu hệ thống (Prerequisites)
Bạn cần cài đặt Python 3.8+ và các thư viện cần thiết. Chạy lệnh sau trong Terminal:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 1. Đào tạo lại mô hình (Re-training) - (Tùy chọn)
Nếu bạn muốn tìm hiểu cách dữ liệu được xử lý hoặc muốn thay đổi thuật toán, hãy mở Jupyter Notebook:
- Khởi động Jupyter Notebook hoặc mở trên VS Code.
- Mở file `evaluation/linear_regression.ipynb`.
- Chạy tất cả các cell (Run All) để xuất ra file model mới lưu vào thư mục `models/`.

### 2. Khởi chạy Ứng dụng Web (Web Deployment)
Để xem giao diện người dùng và tự mình dự đoán giá nhà, hãy khởi động server Streamlit bằng lệnh:
```bash
streamlit run app.py
```
Sau đó, hãy mở trình duyệt web và truy cập vào đường dẫn: **http://localhost:8501**

---

## 📈 Giới thiệu Về Dữ liệu (Dataset)
Bộ dữ liệu sử dụng là [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Kaggle). Với tổng cộng 79 biến giải thích (explanatory variables) mô tả gần như mọi khía cạnh của một căn nhà ở khu dân cư Ames, Iowa.

> **Lưu ý:** Việc áp dụng Log Transform cho biến `SalePrice` là cần thiết để triệt tiêu độ lệch phải (right-skewed) của giá nhà trong tự nhiên.

---
<div align="center">
  <h2>📬 Liên hệ (Contact Information)</h2>
  
  <table>
    <tr>
      <td>👩🏻‍💻 <b>Tác giả:</b></td>
      <td><b>Nguyễn Thị Tú Viên</b></td>
    </tr>
    <tr>
      <td>📧 <b>Email:</b></td>
      <td><a href="mailto:nguyenthituvien2005@gmail.com">nguyenthituvien2005@gmail.com</a></td>
    </tr>
    <tr>
      <td>🎓 <b>Trường đại học:</b></td>
      <td>University of Transport in Ho Chi Minh City</td>
    </tr>
  </table>
  
  <p><i>Cảm ơn vì đã ghé thăm repository của mình! Chúc các bạn một ngày code vui vẻ. 💻✨</i></p>
</div>