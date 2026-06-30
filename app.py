import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & ELEGANT LIGHT CSS
# ==========================================
st.set_page_config(page_title="AI House Predictor | Premium", page_icon="🏘️", layout="wide")

custom_css = """
<style>
    /* 1. Thiết lập màu nền sáng Gradient Xanh - Tím nhạt siêu mịn */
    .stApp {
        background: linear-gradient(135deg, #F3F4F6 0%, #EEF2FF 50%, #E0E7FF 100%) !important;
        color: #1E293B !important;
    }
    
    /* 2. Đảm bảo chữ của các thành phần mặc định có màu tối dễ đọc */
    .stApp label, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1E1B4B !important; /* Màu xanh đen đậm quý phái */
    }

    /* 3. Tiêu đề hiệu ứng Gradient Indigo - Blue sang trọng */
    .title-text {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        color: #475569 !important;
        font-size: 1.1rem;
        margin-bottom: 35px;
    }
    
    /* 4. Định hình lại Form nhập liệu thành dạng Card trắng tinh tế */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.04) !important;
        padding: 30px !important;
    }

    /* 5. Nút bấm với dải màu Tím - Xanh Chàm sang trọng */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.35) !important;
    }

    /* 6. Card hiển thị kết quả cao cấp tông Trắng - Viền Tím */
    .premium-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 6px solid #4F46E5 !important;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.05) !important;
    }
    .price-tag {
        font-size: 3.2rem;
        font-weight: 800;
        color: #4F46E5; /* Màu xanh tím Indigo */
        margin: 10px 0;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #64748B !important;
    }

    /* 7. Định dạng lại các ô nhập liệu cho đồng bộ màu nền sáng */
    .stApp input, .stApp select, .stApp div[role="combobox"] {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('models/linear_regression_model.pkl')

@st.cache_data
def load_base_data():
    return pd.read_csv('data/train.csv')

try:
    model = load_model()
    df = load_base_data()
except Exception as e:
    st.error("Không thể tải Model hoặc Dataset. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()

base_house = df.drop(columns=['Id', 'SalePrice'], errors='ignore').iloc[0].copy()
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if col not in ['Id', 'SalePrice']:
        base_house[col] = df[col].median()
for col in df.select_dtypes(include=['object']).columns:
    base_house[col] = df[col].mode()[0]

average_price = df['SalePrice'].median()

# ==========================================
# 3. HEADER
# ==========================================
st.markdown('<div class="title-text">🏘️ AI Smart Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Hệ thống định giá Bất Động Sản tự động dựa trên thuật toán Học Máy tiên tiến</div>', unsafe_allow_html=True)

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.subheader("🎛️ Thông số thuộc tính")
    
    with st.form("input_form"):
        st.write("Vui lòng thiết lập các thông số chính bên dưới:")
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            gr_liv_area = st.number_input("Tổng diện tích ở (sqft)", min_value=300, max_value=5000, value=1500, step=50)
            overall_qual = st.slider("Chất lượng hoàn thiện (1-10)", min_value=1, max_value=10, value=6)
            garage_cars = st.selectbox("Sức chứa Garage (Số xe)", options=[0, 1, 2, 3, 4], index=2)
            
        with col_i2:
            year_built = st.number_input("Năm xây dựng", min_value=1800, max_value=2026, value=2000, step=1)
            total_bsmt_sf = st.slider("Diện tích tầng hầm (sqft)", min_value=0, max_value=3000, value=1000)
            
        with st.expander("🛠️ Cấu hình vị trí (Nâng cao)"):
            neighborhoods = sorted(df['Neighborhood'].unique())
            default_neighbor_idx = int(np.where(df['Neighborhood'].unique() == base_house['Neighborhood'])[0][0])
            neighborhood = st.selectbox("Khu vực lân cận (Neighborhood)", neighborhoods, index=default_neighbor_idx)

        base_house['OverallQual'] = overall_qual
        base_house['GrLivArea'] = gr_liv_area
        base_house['YearBuilt'] = year_built
        base_house['TotalBsmtSF'] = total_bsmt_sf
        base_house['GarageCars'] = garage_cars
        base_house['Neighborhood'] = neighborhood

        input_df = pd.DataFrame([base_house])
        
        submit_btn = st.form_submit_button("🚀 XEM ĐỊNH GIÁ NGAY")

# ==========================================
# 5. OUTPUT & ANALYSIS
# ==========================================
with col_output:
    st.subheader("📊 Kết quả & Phân tích")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Kết quả dự đoán", "📈 Phân tích thị trường", "ℹ️ Chi tiết mô hình"])
    
    if submit_btn:
        with st.spinner("Đang tính toán giá trị tài sản..."):
            log_price = model.predict(input_df)[0]
            actual_price = np.expm1(log_price)
            
            price_per_sqft = actual_price / gr_liv_area
            delta = actual_price - average_price
            delta_percent = (delta / average_price) * 100
            
            with tab1:
                # Custom White-Indigo Card
                html_result = f"""
                <div class="premium-card">
                    <p style="margin:0; font-size:1.1rem; color: #475569;">Mức giá ước tính (Estimated Value)</p>
                    <div class="price-tag">${actual_price:,.0f}</div>
                    <p class="metric-sub">Đơn giá ước tính: <strong>${price_per_sqft:,.2f}</strong> / sqft</p>
                </div>
                """
                st.markdown(html_result, unsafe_allow_html=True)
                
                st.markdown("##### So sánh với phân khúc trung bình:")
                m1, m2 = st.columns(2)
                m1.metric("Giá trung vị thị trường", f"${average_price:,.0f}")
                m2.metric("Chênh lệch thị trường", f"${delta:,.0f}", f"{delta_percent:+.1f}%")
                
                if delta > 0:
                    st.success("Tài sản này nằm ở phân khúc **trên trung bình** nhờ sở hữu các thông số diện tích và chất lượng vượt trội.")
                else:
                    st.info("Tài sản này thuộc phân khúc **phổ thông**, tiếp cận tốt với đại đa số người mua nhà.")
            
            with tab2:
                st.markdown("##### Vị thế tài sản của bạn trên bản đồ giá thị trường")
                st.caption("Biểu đồ thể hiện mối tương quan giữa Diện tích ở (GrLivArea) và Giá bán (SalePrice). Điểm màu đỏ đại diện cho ngôi nhà bạn vừa thiết lập.")
                
                plot_df = df[['GrLivArea', 'SalePrice']].copy()
                plot_df['Phân loại'] = 'Thị trường chung'
                
                user_house_row = pd.DataFrame({
                    'GrLivArea': [gr_liv_area],
                    'SalePrice': [actual_price],
                    'Phân loại': ['Ngôi nhà của bạn']
                })
                combined_df = pd.concat([plot_df, user_house_row], ignore_index=True)
                
                st.scatter_chart(
                    combined_df, 
                    x='GrLivArea', 
                    y='SalePrice', 
                    color='Phân loại', 
                    use_container_width=True
                )
            
            with tab3:
                st.markdown("##### Phương pháp luận & Chỉ số mô hình")
                st.write("""
                - **Thuật toán sử dụng:** Hồi quy tuyến tính (Linear Regression) kết hợp Ridge Regularization.
                - **Bộ dữ liệu huấn luyện:** Tập dữ liệu Ames Housing nổi tiếng bao gồm các thuộc tính chi tiết của bất động sản.
                - **Thiết kế hệ thống:** Các thuộc tính mở rộng được tự động điền thông qua giá trị trung vị của phân khúc để đảm bảo tốc độ và trải nghiệm nhập liệu tinh giản nhất.
                """)
                
    else:
        with tab1:
            st.info("👈 Vui lòng thiết lập các thông số bên trái và bấm **🚀 XEM ĐỊNH GIÁ NGAY**.")
            st.markdown("##### Biểu đồ phân phối giá nhà theo chất lượng xây dựng:")
            median_by_qual = df.groupby('OverallQual')['SalePrice'].median().reset_index()
            st.area_chart(median_by_qual.set_index('OverallQual'))
            
        with tab2:
            st.warning("Vui lòng thực hiện dự đoán ở cột bên trái để so sánh trực quan.")
        with tab3:
            st.write("Thông tin chi tiết về mô hình sẽ hiển thị sau khi chạy dự báo.")