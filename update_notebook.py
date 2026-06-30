import json

with open('evaluation/linear_regression.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# 1. Update Imports
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if 'import pandas as pd' in src:
            cell['source'] = [
                src,
                "\nfrom sklearn.linear_model import Ridge, Lasso\n",
                "from sklearn.model_selection import GridSearchCV\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "import joblib\n"
            ]
            break

# 2. Add outlier removal and log transform after loading data
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "df = pd.read_csv('../data/train.csv')" in src:
            cell['source'] = [
                "df = pd.read_csv('../data/train.csv')\n\n",
                "# Xử lý Outlier: Loại bỏ những điểm bất thường (nhà quá to nhưng giá quá rẻ)\n",
                "df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)\n",
                "df = df.reset_index(drop=True)\n\n",
                "# Log transform SalePrice\n",
                "df['SalePrice'] = np.log1p(df['SalePrice'])\n",
                "print(f\"Shape sau khi loại bỏ outlier: {df.shape}\")"
            ]
            break

# 3. Update Model Selection (Step 4) to include Ridge and Lasso
# Find "model.fit(X_train, y_train)" and replace that block
idx_model = -1
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "model = Pipeline" in src and "LinearRegression" in src:
            idx_model = i
            break

if idx_model != -1:
    cells[idx_model]['source'] = [
        "numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns\n",
        "categorical_features = X_train.select_dtypes(include=['object']).columns\n\n",
        "numerical_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())\n",
        "])\n\n",
        "categorical_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),\n",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
        "])\n\n",
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', numerical_transformer, numerical_features),\n",
        "        ('cat', categorical_transformer, categorical_features)\n",
        "    ])\n\n",
        "# 1. Linear Regression\n",
        "model_lr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])\n",
        "model_lr.fit(X_train, y_train)\n",
        "print(\"Linear Regression trained.\")\n\n",
        "# 2. Ridge Regression\n",
        "model_ridge = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge())])\n",
        "param_grid_ridge = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}\n",
        "grid_ridge = GridSearchCV(model_ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')\n",
        "grid_ridge.fit(X_train, y_train)\n",
        "print(f\"Ridge trained. Best alpha: {grid_ridge.best_params_['regressor__alpha']}\")\n\n",
        "# 3. Chọn mô hình tốt nhất (Trong ví dụ này giả sử chọn Ridge là best model)\n",
        "best_model = grid_ridge.best_estimator_\n"
    ]

# 4. Add Step 5: Evaluation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Evaluation & Tuning\n",
        "Sử dụng mô hình tốt nhất để dự đoán trên tập Test. Đánh giá bằng RMSE, MAE, R2."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "y_pred = best_model.predict(X_test)\n\n",
        "# Đưa giá trị về lại gốc do đã dùng log1p\n",
        "y_test_expm1 = np.expm1(y_test)\n",
        "y_pred_expm1 = np.expm1(y_pred)\n\n",
        "rmse = np.sqrt(mean_squared_error(y_test_expm1, y_pred_expm1))\n",
        "mae = mean_absolute_error(y_test_expm1, y_pred_expm1)\n",
        "r2 = r2_score(y_test_expm1, y_pred_expm1)\n\n",
        "print(f\"RMSE: {rmse:,.2f}\")\n",
        "print(f\"MAE: {mae:,.2f}\")\n",
        "print(f\"R2 Score: {r2:.4f}\")"
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize Actual vs Predicted\n",
        "plt.figure(figsize=(12, 6))\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.scatter(y_test_expm1, y_pred_expm1, alpha=0.5)\n",
        "plt.plot([y_test_expm1.min(), y_test_expm1.max()], [y_test_expm1.min(), y_test_expm1.max()], 'r--')\n",
        "plt.xlabel('Thực tế (Actual SalePrice)')\n",
        "plt.ylabel('Dự đoán (Predicted SalePrice)')\n",
        "plt.title('Actual vs Predicted')\n\n",
        "# Visualize Residuals\n",
        "plt.subplot(1, 2, 2)\n",
        "residuals = y_test_expm1 - y_pred_expm1\n",
        "sns.histplot(residuals, kde=True)\n",
        "plt.xlabel('Residuals (Phần dư)')\n",
        "plt.title('Residual Distribution')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# 5. Add Step 6: Deployment
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Deployment & Monitoring\n",
        "Lưu lại (Export) mô hình pipeline bằng joblib để tái sử dụng trên dữ liệu mới mà không cần phải huấn luyện lại."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "os.makedirs('../models', exist_ok=True)\n",
        "joblib.dump(best_model, '../models/linear_regression_model.pkl')\n",
        "print(\"Model saved to ../models/linear_regression_model.pkl\")"
    ]
})

with open('evaluation/linear_regression.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

