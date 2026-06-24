import nbformat as nbf

nb = nbf.v4.new_notebook()

# Define Problem
cell_0 = nbf.v4.new_markdown_cell("""# Linear Learning: House Price Prediction

## Bước 1: Data Collection & Overview

### 1.1 Define Problem
Mục tiêu của dự án này là xây dựng một mô hình **Linear Regression** cơ bản để dự đoán giá nhà (`SalePrice`) dựa trên các đặc trưng (features) của ngôi nhà.
Đây là một bài toán **Hồi quy (Regression)** điển hình. Quá trình này được xây dựng bám sát theo luồng E2E từ lúc nhận dữ liệu đến khi đánh giá mô hình.""")

# E2E Processing Flow
cell_e2e = nbf.v4.new_markdown_cell("""### 1.2 E2E Processing Flow
Dưới đây là luồng xử lý tổng thể (End-to-End Pipeline) của bài toán, từ lúc thu thập dữ liệu thô cho đến khi mô hình được huấn luyện:
1. **Data Collection & Overview**: Nhận và hiểu dữ liệu thô.
2. **Exploratory Data Analysis (EDA)**: Khám phá phân phối dữ liệu và tìm ra các đặc trưng có ảnh hưởng mạnh nhất tới giá nhà.
3. **Data Cleaning & Preprocessing**: Làm sạch nhiễu (outliers), xử lý dữ liệu bị thiếu (missing values) và chuyển đổi các biến dạng chữ thành số (One-Hot Encoding).
4. **Train/Test Split**: Chia nhỏ tập dữ liệu để huấn luyện và kiểm thử.
5. **Linear Regression Model**: Lựa chọn và huấn luyện mô hình Linear Regression.
6. **Evaluation & Conclusion**: Đánh giá hiệu suất của mô hình (bằng các chỉ số MAE, RMSE, R²) và đưa ra kết luận.

*(Tham khảo thêm sơ đồ khối tại `../image/image_flowchart.png`)*""")

# Import Libraries
cell_1_md = nbf.v4.new_markdown_cell("""### 1.3 Import Libraries
Import các thư viện cần thiết cho việc thao tác dữ liệu (`pandas`, `numpy`), trực quan hóa (`matplotlib`, `seaborn`) và xây dựng mô hình (`scikit-learn`).""")

cell_1_code = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration for plotting
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")""")

# Load Dataset
cell_2_md = nbf.v4.new_markdown_cell("""### 1.4 Load Dataset
Đọc dữ liệu từ file `train.csv`. Vì mục tiêu là học và tự validate, chúng ta sẽ chỉ dùng tập `train.csv` và sau đó tự chia tập train/test (Hold-out validation).""")
cell_2_code = nbf.v4.new_code_cell("""# Load data
df = pd.read_csv('../data/train.csv')
print(f"Dataset shape: {df.shape}")""")

# Data Overview
cell_3_md = nbf.v4.new_markdown_cell("""### 1.5 Data Overview
Xem xét cấu trúc của tập dữ liệu: các cột, kiểu dữ liệu, và một vài thống kê mô tả cơ bản.""")
cell_3_code = nbf.v4.new_code_cell("""# Display first 5 rows
display(df.head())

# Display basic information
df.info()

# Basic statistics for numerical columns
display(df.describe())""")

# Exploratory Data Analysis
cell_5_md = nbf.v4.new_markdown_cell("""## Bước 2: Exploratory Data Analysis (EDA)

### 2.1 Khám phá Phân phối và Tương quan
Phân tích khám phá dữ liệu:
- Xem phân phối của `SalePrice`.
- Liệt kê top 10 đặc trưng dạng số tương quan mạnh nhất với `SalePrice`.
- Trực quan hóa (Scatter plot và Boxplot) để thấy rõ mối quan hệ.""")
cell_5_1_code = nbf.v4.new_code_cell("""# Distribution of SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, color='blue', bins=30)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()""")

cell_5_2_code = nbf.v4.new_code_cell("""# Top 10 numerical features correlated with SalePrice
numeric_cols = df.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
top_corr_cols = corr_matrix.nlargest(11, 'SalePrice')['SalePrice']
print("--- Top 10 numerical features correlated with SalePrice ---")
display(top_corr_cols.to_frame())

top_corr_matrix = df[top_corr_cols.index].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Top 10 features correlation heatmap')
plt.show()""")

cell_5_3_code = nbf.v4.new_code_cell("""# Scatter plots for key features - Part 1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(ax=axes[0], x=df['OverallQual'], y=df['SalePrice'], alpha=0.6)
axes[0].set_title('OverallQual vs SalePrice')

sns.scatterplot(ax=axes[1], x=df['GrLivArea'], y=df['SalePrice'], alpha=0.6)
axes[1].set_title('GrLivArea vs SalePrice')

sns.scatterplot(ax=axes[2], x=df['TotalBsmtSF'], y=df['SalePrice'], alpha=0.6)
axes[2].set_title('TotalBsmtSF vs SalePrice')
plt.tight_layout()
plt.show()

# Scatter plots for key features - Part 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(ax=axes[0], x=df['YearBuilt'], y=df['SalePrice'], alpha=0.6)
axes[0].set_title('YearBuilt vs SalePrice')

sns.scatterplot(ax=axes[1], x=df['GarageCars'], y=df['SalePrice'], alpha=0.6)
axes[1].set_title('GarageCars vs SalePrice')
plt.tight_layout()
plt.show()""")

cell_5_4_code = nbf.v4.new_code_cell("""# Boxplot SalePrice by OverallQual
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette='Set3')
plt.title('SalePrice grouped by OverallQual')
plt.show()""")

# Data Cleaning
cell_4_md = nbf.v4.new_markdown_cell("""## Bước 3: Data Cleaning & Preprocessing

### 3.1 Data Cleaning
Làm sạch dữ liệu bao gồm:
1. **Loại bỏ dòng trùng lặp**.
2. **Phân tích dữ liệu thiếu (Missing Value Analysis)**: Lập bảng thống kê các cột bị thiếu dữ liệu.
3. **Phân tích ngoại lệ (Outlier Analysis)**: Loại bỏ các điểm dị biệt quá mức (ví dụ: diện tích cực lớn nhưng giá rất rẻ) đã phát hiện trong phần EDA.""")
cell_4_code = nbf.v4.new_code_cell("""# 1. Drop duplicate rows if any
initial_rows = df.shape[0]
df = df.drop_duplicates()
print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.\\n")

# 2. Missing Value Analysis
missing_counts = df.isnull().sum()
missing_percentage = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percentage': missing_percentage})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

print("--- Missing Values Table ---")
display(missing_df)

# 3. Outlier Analysis
# Remove basic outliers: GrLivArea > 4000 & SalePrice < 300000
outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
df = df.drop(outliers)
print(f"\\nDropped {len(outliers)} outlier rows.")""")

# Data Preprocessing
cell_6_md = nbf.v4.new_markdown_cell("""### 3.2 Data Preprocessing
> **Lưu ý về Data Leakage**: Bản notebook này được thiết kế theo hướng "Beginner-friendly", do đó mình sẽ tiền xử lý toàn bộ tập dữ liệu (fill missing values, encode) rồi mới chia X/y. 
> Trong thực tế dự án chuyên nghiệp, để chống rò rỉ dữ liệu (Data Leakage), chúng ta cần chia Train/Test trước, dùng tập Train để fit các transformers (như Imputer) rồi mới transform lên tập Test.

Các bước:
1. Bỏ cột `Id` và các cột thiếu quá nhiều dữ liệu (>80%).
2. Tách `X` (features) và `y` (target).
3. Fill missing values (Median cho số, 'None' cho chữ).
4. Mã hóa Categorical variables thành số bằng One-Hot Encoding.""")
cell_6_code = nbf.v4.new_code_cell("""# 1. Drop ID and columns with > 80% missing values
df = df.drop(columns=['Id'])
missing_ratio = df.isnull().sum() / len(df)
cols_to_drop = missing_ratio[missing_ratio > 0.8].index
df = df.drop(columns=cols_to_drop)
print(f"Dropped columns with too many nulls: {list(cols_to_drop)}")

# 2. Separate features (X) and target (y)
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# 3. Fill missing values
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

# Fill numerical with median
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Fill categorical with 'None'
for col in cat_cols:
    X[col] = X[col].fillna('None')

# 4. One-Hot Encoding for categorical features
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"Shape of X after preprocessing: {X.shape}")""")

# Train/Test Split
cell_7_md = nbf.v4.new_markdown_cell("""## Bước 4: Train/Test Split

### 4.1 Chia Tập Huấn luyện & Kiểm thử
Chia `X` và `y` thành tập Huấn luyện (80%) và tập Kiểm tra (20%).""")
cell_7_code = nbf.v4.new_code_cell("""# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")""")

# Build Linear Regression Model
cell_8_md = nbf.v4.new_markdown_cell("""## Bước 5: Linear Regression Model

### 5.1 Build Linear Regression Model
Khởi tạo và huấn luyện (fit) mô hình Linear Regression cơ bản nhất.""")
cell_8_code = nbf.v4.new_code_cell("""# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Linear Regression Model successfully trained!")""")

# Model Evaluation
cell_9_md = nbf.v4.new_markdown_cell("""## Bước 6: Evaluation & Conclusion

### 6.1 Model Evaluation
Dự đoán trên tập `X_test` và tính toán các độ đo: **MAE**, **MSE**, **RMSE**, **R² Score**.""")
cell_9_code = nbf.v4.new_code_cell("""# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae, mse, rmse, r2]
})
display(metrics_df)

# Visualize Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Sale Price')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.show()

# Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.show()""")

# Sample Prediction
cell_10_md = nbf.v4.new_markdown_cell("""### 6.2 Sample Prediction
Hiển thị một vài kết quả dự đoán và so sánh trực tiếp với giá trị thực tế.""")
cell_10_code = nbf.v4.new_code_cell("""# Compare actual vs predicted for the first 10 houses in the test set
sample_results = pd.DataFrame({
    'Actual Price': y_test[:10].values,
    'Predicted Price': np.round(y_pred[:10], 2),
    'Difference': np.round(y_test[:10].values - y_pred[:10], 2)
})
display(sample_results)""")

# Conclusion
cell_11_md = nbf.v4.new_markdown_cell("""### 6.3 Conclusion
**Nhận xét về Baseline Model (Linear Regression):**
1. **R² Score** đạt khoảng **0.8805** (88.05%), chứng tỏ các biến đặc trưng giải thích được phần lớn sự biến động của giá nhà. Đây là một kết quả tốt đối với một mô hình tuyến tính cơ bản.
2. **RMSE** thực tế đạt khoảng **25.690 USD**. Việc này cho thấy vẫn còn độ lệch tương đối ở một số dự đoán (đặc biệt với các căn nhà giá trị cao).
3. Nhìn vào **Residual Plot**, sai số tăng dần theo giá trị thực tế của căn nhà. Điều này cho thấy Linear Regression bị ảnh hưởng nhiều bởi các ngoại lệ và không hoàn toàn bắt được các mối quan hệ phi tuyến tính.

**Hướng cải thiện (Future Improvement):**
- Xử lý thêm các biến lệch (Skewed variables) bằng phép biến đổi logarit (log-transform).
- Sử dụng **Ridge Regression** hoặc **Lasso Regression** để kiểm soát các hệ số (Regularization), chống Overfitting khi ta đang dùng quá nhiều cột sinh ra từ One-Hot Encoding.""")

nb['cells'] = [
    cell_0, cell_e2e, cell_1_md, cell_1_code,
    cell_2_md, cell_2_code,
    cell_3_md, cell_3_code,
    cell_5_md, cell_5_1_code, cell_5_2_code, cell_5_3_code, cell_5_4_code,
    cell_4_md, cell_4_code,
    cell_6_md, cell_6_code,
    cell_7_md, cell_7_code,
    cell_8_md, cell_8_code,
    cell_9_md, cell_9_code,
    cell_10_md, cell_10_code,
    cell_11_md
]

with open('/Users/vientu/AI2026/Linear Learning/evaluation/linear_learning.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
