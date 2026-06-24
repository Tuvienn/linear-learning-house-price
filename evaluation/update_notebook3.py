import json

filepath = "/Users/vientu/AI2026/Linear Learning/evaluation/linear_learning.ipynb"
with open(filepath, 'r') as f:
    nb = json.load(f)

# Cell 3: Add Pipeline and preprocessing imports
code_cell_3 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration for plotting
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")"""

# Markdown Cell 12: Add explanation about data leakage
md_cell_12 = """### Feature Preprocessing
> **Lưu ý về Data Leakage**: Ở bước này, chúng ta chỉ tách `X` và `y`, loại bỏ các cột không cần thiết (như `Id` hoặc cột thiếu quá nhiều dữ liệu).
> **Vì sao phải split train/test trước khi preprocessing?** Nếu chúng ta xử lý dữ liệu (ví dụ: điền missing values bằng trung vị hoặc min/max scaling) trên toàn bộ dữ liệu trước khi chia train/test, thông tin từ tập test sẽ "rò rỉ" (leak) vào tập train. Điều này làm cho mô hình đánh giá sai hiệu suất thực tế. Do đó, bước tiền xử lý sẽ được dời xuống sau khi đã chia tập dữ liệu và sử dụng `Pipeline`.

Các bước:
1. Tách `X` (features) và `y` (target).
2. Bỏ cột `Id` và các cột thiếu quá nhiều dữ liệu (>80%)."""

# Code Cell 13: X, y separation
code_cell_13 = """# 1. Separate features (X) and target (y)
y = df['SalePrice']
X = df.drop(columns=['SalePrice'])

# 2. Drop ID and columns with > 80% missing values
if 'Id' in X.columns:
    X = X.drop(columns=['Id'])

missing_ratio = X.isnull().sum() / len(X)
cols_to_drop = missing_ratio[missing_ratio > 0.8].index
X = X.drop(columns=cols_to_drop)
print(f"Dropped columns with too many nulls: {list(cols_to_drop)}")
print(f"Shape of X: {X.shape}")"""

# Markdown Cell 14 is step 3: Data splitting. No change needed or minor update.
md_cell_14 = """## 3. Data Splitting
Chia `X` và `y` thành tập Huấn luyện (80%) và tập Kiểm tra (20%).
Sử dụng `train_test_split` với `test_size=0.2` và `random_state=42`."""

# Code Cell 15: train_test_split
code_cell_15 = """# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")"""

# Markdown Cell 16: Explain pipeline
md_cell_16 = """## 4. Model Selection & Training
Xây dựng một **Pipeline** bao gồm các bước xử lý dữ liệu và huấn luyện mô hình.
1. **Xác định loại biến**: Chia các cột thành số (numerical) và chữ (categorical) dựa trên `X_train`.
2. **Tạo Preprocessing Pipeline**:
   - *Numerical*: Lấp đầy giá trị thiếu bằng trung vị (`SimpleImputer(strategy='median')`), sau đó chuẩn hóa (`StandardScaler()`).
   - *Categorical*: Lấp đầy giá trị thiếu bằng chuỗi 'None' (`SimpleImputer(strategy='constant')`), sau đó mã hóa (`OneHotEncoder(handle_unknown='ignore')`).
3. **Full Pipeline**: Kết hợp preprocessing bằng `ColumnTransformer` và mô hình `LinearRegression`.

> **Vì sao fit preprocessing chỉ trên training set?** Để đảm bảo tính khách quan tuyệt đối. Các transformer (như Imputer, Scaler) chỉ được học (fit) các tham số (ví dụ: mean, std, median) từ tập train. Khi áp dụng cho tập test, nó chỉ transform dựa trên những gì đã học từ tập train.
> **Vì sao OneHotEncoder cần handle_unknown='ignore'?** Trong tập test có thể xuất hiện những category mới (chưa từng thấy trong tập train). Tham số này giúp bỏ qua category đó thay vì báo lỗi.
> **Vì sao dùng Pipeline?** Pipeline đóng gói toàn bộ quy trình preprocessing và model thành một khối, đảm bảo không bao giờ bị Data Leakage và dễ dàng triển khai (deploy)."""

# Code cell 17: Pipeline and fit
code_cell_17 = """# 1. Identify numerical and categorical columns from X_train
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# 2. Create preprocessing pipelines for both types of features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Create the full pipeline with preprocessing and the linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Train the model using the pipeline
model.fit(X_train, y_train)
print("Pipeline with Linear Regression Model successfully trained!")"""

# Code cell 21: Sample prediction using pipeline
code_cell_21 = """# Take a single sample from the test set for prediction
sample_house = X_test.iloc[[0]]
actual_price = y_test.iloc[0]

# Predict using the full pipeline
predicted_price = model.predict(sample_house)[0]

print("--- Sample Prediction ---")
display(sample_house)
print(f"Actual Price:    ${actual_price:,.2f}")
print(f"Predicted Price: ${predicted_price:,.2f}")
print(f"Difference:      ${actual_price - predicted_price:,.2f}")"""


def update_cell(cell, new_content):
    lines = new_content.split('\n')
    cell["source"] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]


for idx, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] == "code":
        if idx == 3: update_cell(cell, code_cell_3)
        elif idx == 13: update_cell(cell, code_cell_13)
        elif idx == 15: update_cell(cell, code_cell_15)
        elif idx == 17: update_cell(cell, code_cell_17)
        elif idx == 21: update_cell(cell, code_cell_21)
    elif cell["cell_type"] == "markdown":
        if idx == 12: update_cell(cell, md_cell_12)
        elif idx == 14: update_cell(cell, md_cell_14)
        elif idx == 16: update_cell(cell, md_cell_16)

with open(filepath, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with scikit-learn Pipeline!")
