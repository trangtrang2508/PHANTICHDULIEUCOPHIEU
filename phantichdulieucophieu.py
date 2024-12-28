import streamlit as st  
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import warnings
import plotly.graph_objects as go
import altair as alt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# Cài đặt Streamlit
st.set_page_config(page_title="PHÂN TÍCH DỮ LIỆU CỔ PHẾU", layout="wide", initial_sidebar_state="expanded")
st.title("PHÂN TÍCH DỮ LIỆU CỔ PHẾU")

# Khởi tạo danh sách mã cổ phiếu và tên công ty
tickers = ['MSFT', 'ORCL', 'IBM', 'CRM', 'SAP']
company_names = {'MSFT': 'Microsoft', 'ORCL': 'Oracle', 'IBM': 'IBM', 'CRM': 'Salesforce', 'SAP': 'SAP'}

# Sidebar: Bộ lọc
st.sidebar.title("Bộ lọc")
selected_ticker = st.sidebar.selectbox("Chọn công ty:", tickers)
selected_company = company_names[selected_ticker]
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", datetime.now())
st.sidebar.markdown(
    """
    <div style="background-color:#2E3B4E; padding:10px; border-radius:5px; text-align:center;">
        <p style="color:#FFFFFF; font-size:14px; margin:0;">
            Created and designed by NghiemThiNgocThao and NgoNguyenAnhTrang 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tải dữ liệu cổ phiếu
stock_data = yf.download(selected_ticker, start=start_date, end=end_date)
stock_data = stock_data[~stock_data.applymap(lambda x: str(x).strip()).isin(['1'])]
st.write("DỮ LIỆU GỐC", stock_data.head(20))

# Đảm bảo index là Datetime
if not isinstance(stock_data.index, pd.DatetimeIndex):
    stock_data.index = pd.to_datetime(stock_data.index)

# Kiểm tra cột 'Close'
if 'Close' not in stock_data.columns:
    st.error("Dữ liệu không chứa cột 'Close'. Kiểm tra lại nguồn dữ liệu.")
    st.stop()

# Kiểm tra dữ liệu
if stock_data.empty:
    st.error("Không có dữ liệu nào được tải. Hãy chọn khoảng thời gian hợp lệ.")
else:
    # Chuẩn bị dữ liệu chung
    stock_data['Year'] = stock_data.index.year
    stock_data['Quarter'] = stock_data.index.quarter
    stock_data_reset = stock_data.reset_index()

    # Tabs: Thống kê, Phân tích, Mô hình dự báo
    tab1, tab2, tab3 = st.tabs(["THỐNG KÊ MÔ TẢ", "CHUỖI THỜI GIAN", "MÔ HÌNH DỰ BÁO"])

    # ----------- Tab 1: Thống kê mô tả ----------- #
    with tab1:
        # Chọn chỉ các cột số
        numeric_columns = stock_data.select_dtypes(include=['float64', 'int64'])
        # Thực hiện thống kê trên các cột số
        statistics = numeric_columns.aggregate(['min', 'max', 'mean', 'median', 'std'])
        # Hiển thị kết quả thống kê trên Streamlit
        title=f"THỐNG KÊ DỮ LIỆU {selected_company}",
        st.write(statistics)
        
        # Biểu đồ giá đóng theo ngày
        price_chart = alt.Chart(stock_data_reset).mark_line().encode(
                x=alt.X('Date:T', title='Ngày', axis=alt.Axis(format='%Y-%m-%d')),
                y=alt.Y('Close:Q', title='Giá đóng (USD)'),
                tooltip=[alt.Tooltip('Date:T', title='Ngày', format='%Y-%m-%d'),
                         alt.Tooltip('Close:Q', title='Giá đóng', format=',.2f')]
            ).properties(
                title=f"Giá đóng của {selected_company} theo ngày",
                width=800,
                height=400
            )
            
        st.altair_chart(price_chart, use_container_width=True)

        # Trung bình giá đóng cửa mỗi năm
        stock_data['Year'] = stock_data['Year'].apply(lambda x: int(str(x).replace(',', '')))

        avg_close_by_year = stock_data.groupby('Year')['Close'].mean().reset_index()

        avg_close_chart = alt.Chart(avg_close_by_year).mark_bar().encode(
                x=alt.X('Year:O', title='Năm'),
                y=alt.Y('Close:Q', title='Giá trung bình (USD)'),
                tooltip=[alt.Tooltip('Year:O', title='Năm'),
                         alt.Tooltip('Close:Q', title='Giá trung bình', format=',.2f')]
            ).properties(
                title="Trung bình giá đóng cửa mỗi năm",
                width=800,
                height=400
            )
            
        st.altair_chart(avg_close_chart, use_container_width=True)

        # Trung bình giá đóng cửa theo quý
        avg_close_by_quarter = stock_data.groupby(['Year', 'Quarter'])['Close'].mean().reset_index()
        quarter_chart = alt.Chart(avg_close_by_quarter).mark_bar().encode(
                x=alt.X('Year:O', title='Năm'),
                y=alt.Y('Close:Q', title='Giá trung bình (USD)'),
                color=alt.Color('Quarter:O', title='Quý'),
                tooltip=[alt.Tooltip('Year:O', title='Năm'),
                         alt.Tooltip('Quarter:O', title='Quý'),
                         alt.Tooltip('Close:Q', title='Giá trung bình', format=',.2f')]
            ).properties(
                title="Trung bình giá đóng cửa theo quý",
                width=800,
                height=400
            )
            
        st.altair_chart(quarter_chart, use_container_width=True)

        # Biểu đồ khối lượng giao dịch
        volume_chart = alt.Chart(stock_data_reset).mark_line(color='green').encode(
                x=alt.X('Date:T', title='Ngày', axis=alt.Axis(format='%Y-%m-%d')),
                y=alt.Y('Volume:Q', title='Khối lượng giao dịch', axis=alt.Axis(format=',.0f')),
                tooltip=[alt.Tooltip('Date:T', title='Ngày', format='%Y-%m-%d'),
                         alt.Tooltip('Volume:Q', title='Khối lượng giao dịch', format=',.0f')]
            ).properties(
                title=f"Khối lượng giao dịch của {selected_company}",
                width=800,
                height=400
            )
        
        st.altair_chart(volume_chart, use_container_width=True)

        # Tổng lượng cổ phiếu giao dịch mỗi năm
        volume_by_year = stock_data.groupby('Year')['Volume'].sum().reset_index()

        volume_by_year_chart = alt.Chart(volume_by_year).mark_bar(color='green').encode(
                x=alt.X('Year:O', title='Năm'),
                y=alt.Y('Volume:Q', title='Tổng lượng giao dịch', axis=alt.Axis(format=',.0f')),
                tooltip=[alt.Tooltip('Year:O', title='Năm'),
                         alt.Tooltip('Volume:Q', title='Tổng lượng giao dịch', format=',.0f')]
            ).properties(
                title="Tổng lượng giao dịch cổ phiếu mỗi năm",
                width=800,
                height=400
            )
            
        st.altair_chart(volume_by_year_chart, use_container_width=True)

        # Tổng lượng cổ phiếu giao dịch theo quý
        sum_volume_by_quarter = stock_data.groupby(['Year', 'Quarter'])['Volume'].sum().reset_index()
            
        volume_by_quarter_chart = alt.Chart(sum_volume_by_quarter).mark_bar().encode(
                x=alt.X('Quarter:O', title='Quý'),
                y=alt.Y('Volume:Q', title='Lượng cổ phiếu giao dịch', axis=alt.Axis(format='~s')),
                color=alt.Color('Year:N', title='Năm'),
                tooltip=[alt.Tooltip('Quarter:O', title='Quý'),
                         alt.Tooltip('Year:N', title='Năm'),
                         alt.Tooltip('Volume:Q', title='Lượng cổ phiếu giao dịch', format=',')]
            ).properties(
                title="Tổng lượng giao dịch cổ phiếu theo quý",
                width=800,
                height=400
            )
            
        st.altair_chart(volume_by_quarter_chart, use_container_width=True)

    # ----------- Tab 2: Chuỗi thời gian ----------- #
    with tab2:
        st.subheader("Phân tích chuỗi thời gian")
        
        if st.checkbox("Thực hiện phân rã chuỗi thời gian"):
            try:
                # Thực hiện phân rã chuỗi thời gian
                result = seasonal_decompose(stock_data['Close'], model='additive', period=365)

                # Các thành phần phân rã
                components = [
                    ('Observed', result.observed, 'Biểu đồ gốc (Observed)'),
                    ('Trend', result.trend, 'Xu hướng (Trend)'),
                    ('Seasonality', result.seasonal, 'Tính mùa vụ (Seasonality)'),
                    ('Residuals', result.resid, 'Phần dư (Residuals)')
                ]

                for component_name, component_data, chart_title in components:
                    # Tạo DataFrame chứa dữ liệu ngày và năm
                    component_df = pd.DataFrame({
                        'Date': stock_data.index,
                        component_name: component_data
                    }).reset_index(drop=True)

                    # Vẽ biểu đồ với trục x định dạng hiển thị tháng trước năm
                    chart = alt.Chart(component_df).mark_line().encode(
                        x=alt.X('Date:T', title='Tháng - Năm', axis=alt.Axis(format='%b %Y', labelAngle=-45)),  
                        # %b %Y hiển thị tháng (tên viết tắt) và năm
                        y=alt.Y(f'{component_name}:Q', title=chart_title),
                        tooltip=[
                            alt.Tooltip('Date:T', title='Ngày'),
                            alt.Tooltip(f'{component_name}:Q', title=chart_title, format=',.2f')
                        ]
                    ).properties(
                        title=chart_title,
                        width=800,
                        height=300
                    )

                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi khi phân rã chuỗi thời gian: {e}")

        # Phân tích tự tương quan
        if st.checkbox("Hiển thị tự tương quan của phần dư"):
            try:
                # Kiểm tra nếu kết quả phân rã đã tồn tại
                if 'result' not in locals():
                    st.warning("Vui lòng thực hiện phân rã chuỗi thời gian trước.")
                else:
                    # Tính toán ACF của phần dư
                    residuals = result.resid.dropna()  # Loại bỏ giá trị NaN
                    acf_values = acf(residuals, nlags=50, fft=True)
                    acf_df = pd.DataFrame({
                        'Lag': range(len(acf_values)),
                        'ACF': acf_values
                    })

                    # Vẽ biểu đồ ACF với Altair
                    acf_chart = alt.Chart(acf_df).mark_bar().encode(
                        x=alt.X('Lag:O', title='Độ trễ (Lag)'),
                        y=alt.Y('ACF:Q', title='Tự tương quan (ACF)', scale=alt.Scale(domain=[-1, 1])),
                        tooltip=[
                            alt.Tooltip('Lag:O', title='Độ trễ'),
                            alt.Tooltip('ACF:Q', title='Tự tương quan', format=',.2f')
                        ]
                    ).properties(
                        title="Tự tương quan của phần dư",
                        width=800,
                        height=400
                    )

                    # Thêm đường tham chiếu
                    line_base = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red').encode(y='y:Q')
                    st.altair_chart(acf_chart + line_base, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi khi phân tích tự tương quan: {e}")

    # ----------- Tab 3: Mô hình dự báo ----------- #
    with tab3:
        st.subheader("Mô hình dự báo")

        # Lựa chọn khung thời gian dự báo
        st.write("Lựa chọn khung thời gian dự báo")
        forecast_type = st.radio(
            "Chọn khung thời gian dự báo:",
            ('Ngày', 'Tuần', 'Tháng'),
            horizontal=True
        )

        # Xử lý dữ liệu theo khung thời gian được chọn
        if forecast_type == 'Ngày':
            data_resampled = stock_data['Close']  # Giữ nguyên dữ liệu hằng ngày
            forecast_steps = 1  # Dự báo 1 ngày tiếp theo
            st.write("Dự báo giá cổ phiếu hằng ngày.")
        elif forecast_type == 'Tuần':
            data_resampled = stock_data['Close'].resample('W').mean()  # Dữ liệu trung bình tuần
            forecast_steps = 7  # Dự báo cho 7 ngày tiếp theo (tuần)
            st.write("Dự báo giá cổ phiếu hằng tuần.")
        elif forecast_type == 'Tháng':
            data_resampled = stock_data['Close'].resample('M').mean()  # Dữ liệu trung bình tháng
            forecast_steps = 30  # Dự báo 30 ngày tiếp theo (tháng)
            st.write("Dự báo giá cổ phiếu hằng tháng.")
        
        # Tùy chọn sử dụng Moving Average
        if st.checkbox("Moving Average"):
            st.write("### Lựa chọn Moving Average:")
            ma_option = st.radio("Chọn loại MA:", ["Naive", "3-Step", "6-Step"])

            # Xác định cửa sổ tương ứng
            if ma_option == "Naive":
                window_size = 1
            elif ma_option == "3-Step":
                window_size = 3
            elif ma_option == "6-Step":
                window_size = 6

            # Tính toán Moving Average
            MA_values = data_resampled.rolling(window=window_size).mean()

            # Loại bỏ NaN khỏi cả data_resampled và MA_values
            valid_indices = MA_values.dropna().index

            # Chuyển đổi dữ liệu thành mảng 1 chiều
            ma_df = pd.DataFrame({
                'Date': data_resampled.loc[valid_indices].index.to_list(),
                'Close': data_resampled.loc[valid_indices].values.flatten(),
                'Moving Average': MA_values.dropna().values.flatten()
            }).reset_index(drop=True)

            # Dự đoán giá tương lai dựa trên xu hướng gần nhất (sử dụng Gradient)
            last_ma_values = MA_values[-window_size:].dropna()
            gradient = (last_ma_values.iloc[-1] - last_ma_values.iloc[0]) / max((window_size - 1), 1)  # Tránh chia cho 0
            forecast_values = [last_ma_values.iloc[-1] + gradient * i for i in range(1, forecast_steps + 1)]

            # Tạo ngày dự báo
            future_dates = [end_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]

            # Chuyển đổi ngày thành chuỗi để tránh lỗi pyarrow
            ma_df['Date'] = ma_df['Date'].dt.strftime('%Y-%m-%d')

            # Thêm dữ liệu Moving Average vào ma_df_melted
            ma_df_melted = pd.DataFrame({
                'Date': ma_df['Date'].tolist() + ma_df['Date'].tolist() + future_dates,
                'Value': ma_df['Close'].tolist() + ma_df['Moving Average'].tolist() + forecast_values,
                'Type': ['Actual Close Price'] * len(ma_df) + ['Moving Average'] * len(ma_df) + ['Forecast'] * len(future_dates)
            })

            # Biểu đồ Altair với chú thích (legend)
            chart_ma = alt.Chart(ma_df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Ngày'),
                y=alt.Y('Value:Q', title='Giá cổ phiếu', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', title='Loại Dữ Liệu'),
                tooltip=[
                    alt.Tooltip('Date:T', title='Ngày'),
                    alt.Tooltip('Value:Q', title='Giá trị', format=',.2f'),
                    alt.Tooltip('Type:N', title='Loại Dữ Liệu')
                ]
            ).properties(
                width=800, height=400, title=f"Biểu đồ Giá Cổ Phiếu và Moving Average ({ma_option}) với Dự báo"
            )

            # Hiển thị biểu đồ
            st.altair_chart(chart_ma, use_container_width=True)

            # Chuẩn bị dữ liệu cho bảng dự báo
            forecast_df = pd.DataFrame({
                'Ngày': [date.strftime('%Y-%m-%d') for date in future_dates],
                'Giá Dự Đoán': [float(value) for value in forecast_values]  # Đảm bảo chỉ có giá trị float
            })
            st.write("### Bảng dữ liệu dự đoán")
            st.dataframe(forecast_df)

            # Tính toán các chỉ số hiệu suất
            st.write("### Đánh giá hiệu suất dự đoán:")
            y_true = data_resampled.loc[valid_indices].values.flatten()
            y_pred = MA_values.dropna().values.flatten()

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            epsilon = 1e-10  # Tránh chia cho 0
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

            # Hiển thị các chỉ số lỗi
            st.metric("MAE", f"{mae:.2f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAPE", f"{mape:.2f} %")

        # Dự báo bằng Simple Exponential Smoothing (SES)
        if st.checkbox("Simple Exponential Smoothing (SES)"):
            # Hàm tìm alpha tối ưu
            def optimize_alpha(data, alpha_values):
                best_alpha = None
                best_mse = float('inf')
                for alpha in alpha_values:
                    model = SimpleExpSmoothing(data).fit(smoothing_level=alpha, optimized=False)
                    fitted_values = model.fittedvalues
                    mse = mean_squared_error(data, fitted_values)
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha
                return best_alpha, best_mse

            # Tập giá trị alpha để thử nghiệm
            alpha_range = np.linspace(0.01, 1.0, 100)
            
            # Tìm alpha tối ưu
            with st.spinner("Đang tối ưu hóa alpha..."):
                optimal_alpha, optimal_mse = optimize_alpha(stock_data['Close'], alpha_range)
            
            # Áp dụng mô hình với alpha tối ưu
            model = SimpleExpSmoothing(stock_data['Close']).fit(smoothing_level=optimal_alpha, optimized=False)
            SES_opt = model.fittedvalues

            # Tính toán dự báo giá tương lai
            # Dự báo tuyến tính dựa trên giá trị cuối cùng
            last_ses_value = SES_opt.iloc[-1]  # Giá trị SES cuối cùng
            ses_trend = SES_opt.iloc[-1] - SES_opt.iloc[-2]  # Tính xu hướng từ 2 giá trị cuối
            future_dates = [end_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]  # Ngày dự báo
            forecast_values = [last_ses_value + (i * ses_trend) for i in range(1, forecast_steps + 1)]  # Dự báo tuyến tính

            # Làm sạch dữ liệu dự báo
            future_dates_cleaned = [date.strftime('%Y-%m-%d') for date in future_dates]
            forecast_values_cleaned = [float(value) for value in forecast_values]

            # Chuẩn bị dữ liệu để vẽ biểu đồ
            ses_df = pd.DataFrame({
                'Date': stock_data.index.to_list(),  # Ngày từ dữ liệu gốc
                'Close': stock_data['Close'].values.flatten(),  # Giá thực tế
                'SES': SES_opt.values.flatten()  # Giá trị SES
            }).reset_index(drop=True)

            ses_df_melted = pd.DataFrame({
                'Date': ses_df['Date'].tolist() + ses_df['Date'].tolist(),
                'Value': ses_df['Close'].tolist() + ses_df['SES'].tolist(),
                'Type': ['Giá thực tế'] * len(ses_df) + [f'SES (Alpha = {optimal_alpha:.2f})'] * len(ses_df)
            })

            # Biểu đồ Altair
            chart_ses = alt.Chart(ses_df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Ngày'),
                y=alt.Y('Value:Q', title='Giá cổ phiếu', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', title='Dữ liệu'),
                tooltip=[
                    alt.Tooltip('Date:T', title='Ngày'),
                    alt.Tooltip('Value:Q', title='Giá cổ phiếu', format=',.2f'),
                    alt.Tooltip('Type:N', title='Loại')
                ]
            ).properties(
                width=800, height=400, title=f"Biểu đồ SES (Alpha = {optimal_alpha:.2f})"
            )

            # Hiển thị biểu đồ SES
            st.altair_chart(chart_ses, use_container_width=True)

            # Bảng dữ liệu dự báo
            forecast_df = pd.DataFrame({
                'Ngày': future_dates_cleaned,
                'Giá Dự Đoán': forecast_values_cleaned
            })

            st.write("### Bảng dữ liệu dự đoán SES")
            st.dataframe(forecast_df[['Ngày', 'Giá Dự Đoán']])

            # Tính toán các chỉ số lỗi
            epsilon = 1e-10
            y_true = stock_data['Close'].values.flatten()
            SES_opt = pd.Series(SES_opt, index=stock_data.index).values.flatten()

            mae_SES_opt = mean_absolute_error(y_true, SES_opt)
            mse_SES_opt = mean_squared_error(y_true, SES_opt)
            rmse_SES_opt = np.sqrt(mse_SES_opt)
            r2_SES_opt = r2_score(y_true, SES_opt)
            mape_SES_opt = np.mean(np.abs((y_true - SES_opt) / (y_true + epsilon))) * 100

            st.write("**Đánh giá hiệu suất:**")
            st.metric("MAE", f"{mae_SES_opt:.2f}")
            st.metric("MSE", f"{mse_SES_opt:.2f}")
            st.metric("RMSE", f"{rmse_SES_opt:.2f}")
            st.metric("MAPE", f"{mape_SES_opt:.2f} %")

        # Dự báo bằng Holt
        use_holt = st.checkbox("Holt")
        if use_holt:
            # Hàm tìm alpha và beta tối ưu
            def optimize_holt(data, alpha_values, beta_values):
                best_alpha = None
                best_beta = None
                best_mse = float('inf')

                for alpha in alpha_values:
                    for beta in beta_values:
                        model = ExponentialSmoothing(data, trend='add', damped_trend=True).fit(
                            smoothing_level=alpha, smoothing_trend=beta)
                        fitted_values = model.fittedvalues
                        mse = mean_squared_error(data, fitted_values)
                        if mse < best_mse:
                            best_mse = mse
                            best_alpha = alpha
                            best_beta = beta
                return best_alpha, best_beta, best_mse

            # Tập giá trị alpha và beta để thử nghiệm
            alpha_range = np.linspace(0.01, 1.0, 50)  # Giảm số lượng để tăng tốc
            beta_range = np.linspace(0.01, 1.0, 50)

            # Tìm alpha và beta tối ưu
            with st.spinner("Đang tối ưu hóa alpha và beta..."):
                optimal_alpha, optimal_beta, optimal_mse = optimize_holt(stock_data['Close'], alpha_range, beta_range)
            
            # Áp dụng mô hình với alpha và beta tối ưu
            final_model = ExponentialSmoothing(stock_data['Close'], trend='add', damped_trend=True).fit(
                smoothing_level=optimal_alpha, smoothing_trend=optimal_beta)
            Holt_opt = final_model.fittedvalues

            # Chuẩn bị dữ liệu cho Altair
            holt_df = pd.DataFrame({
                'Date': stock_data.index.to_list(),  # Sử dụng chỉ số của stock_data làm 'Date'
                'Close': stock_data['Close'].values.flatten(),  # Chuyển đổi thành mảng 1 chiều
                'Holt': Holt_opt.values.flatten()  # Chuyển đổi thành mảng 1 chiều
            }).reset_index(drop=True)

            # Dự báo với số bước thời gian đã chọn
            forecast_holt = final_model.forecast(steps=forecast_steps)
            forecast_dates = [end_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            forecast_df_holt = pd.DataFrame({"Ngày": forecast_dates, "Dự báo giá": forecast_holt.values})

            # Kiểm tra và chuẩn hóa chiều dài các mảng
            holt_dates = holt_df['Date'].tolist()
            holt_close = holt_df['Close'].tolist()
            holt_holt = holt_df['Holt'].tolist()
            forecast_dates_list = forecast_dates
            forecast_holt_list = forecast_holt.tolist()

            # Đảm bảo các danh sách được tạo khớp
            if len(holt_dates) == len(holt_close) == len(holt_holt) and len(forecast_dates_list) == len(forecast_holt_list):
                holt_df_melted = pd.DataFrame({
                    'Date': holt_dates + holt_dates + forecast_dates_list,
                    'Value': holt_close + holt_holt + forecast_holt_list,
                    'Type': (['Giá thực tế'] * len(holt_dates) +
                            [f'Holt (α={optimal_alpha:.2f}, β={optimal_beta:.2f})'] * len(holt_dates) +
                            ['Dự báo Giá'] * len(forecast_dates_list))
                })
            else:
                st.error("Dữ liệu không khớp độ dài. Vui lòng kiểm tra lại.")

            # Chuyển đổi các cột ngày trước khi vẽ biểu đồ
            holt_df_melted['Date'] = pd.to_datetime(holt_df_melted['Date'])
            forecast_df_holt['Ngày'] = pd.to_datetime(forecast_df_holt['Ngày'])

            # Sửa danh sách ngày dự báo
            forecast_dates = [pd.Timestamp(end_date + timedelta(days=i)) for i in range(1, forecast_steps + 1)]

            # Biểu đồ Altair với chú thích (legend)
            chart_holt = alt.Chart(holt_df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Ngày'),
                y=alt.Y('Value:Q', title='Giá cổ phiếu', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', title='Loại Dữ Liệu'),  # Tạo legend từ cột 'Type'
                tooltip=[
                    alt.Tooltip('Date:T', title='Ngày'),
                    alt.Tooltip('Value:Q', title='Giá cổ phiếu', format=',.2f'),
                    alt.Tooltip('Type:N', title='Loại Dữ Liệu')
                ]
            ).properties(
                width=800, height=400, title=f"Biểu đồ Dự báo với Holt (α={optimal_alpha:.2f}, β={optimal_beta:.2f})"
            )

            # Hiển thị biểu đồ Holt với Altair
            st.altair_chart(chart_holt, use_container_width=True)

            # Hiển thị dự báo
            st.write("Dự báo giá cổ phiếu:")
            st.dataframe(forecast_df_holt)
            
            # Chuyển stock_data['Close'] thành mảng 1 chiều
            y_true = stock_data['Close'].values.flatten()
            # Đồng bộ Holt_opt với chỉ số của stock_data['Close']
            Holt_opt = pd.Series(Holt_opt, index=stock_data.index)

            #  Đảm bảo rằng Holt_opt có cùng chỉ số với y_true
            Holt_opt = Holt_opt.reindex(stock_data.index, method='nearest').values.flatten()
            
            # Tính toán các chỉ số lỗi
            epsilon = 1e-10  # Giá trị rất nhỏ để tránh chia cho 0
            mae_holt_opt = mean_absolute_error(y_true, Holt_opt)
            mse_holt_opt = mean_squared_error(y_true, Holt_opt)
            rmse_holt_opt = np.sqrt(mse_holt_opt)
            mape_holt_opt = np.mean(np.abs((y_true - Holt_opt) / y_true + epsilon)) * 100

            # Hiển thị các chỉ số lỗi
            st.write("**Đánh giá hiệu suất:**")
            st.metric("Optimal Alpha", f"{optimal_alpha:.2f}")
            st.metric("Optimal Beta", f"{optimal_beta:.2f}")
            st.metric("MAE", f"{mae_holt_opt:.2f}")
            st.metric("MSE", f"{mse_holt_opt:.2f}")
            st.metric("RMSE", f"{rmse_holt_opt:.2f}")
            st.metric("MAPE", f"{mape_holt_opt:.2f}%")

        # Dự báo bằng Holt-Winters
        if st.checkbox("Holt Winters"):
            # Hàm tìm alpha, beta, gamma tối ưu
            def optimize_holt_winters(data, alpha_values, beta_values, gamma_values, seasonal_periods):
                best_alpha, best_beta, best_gamma = None, None, None
                best_mse = float('inf')

                for alpha in alpha_values:
                    for beta in beta_values:
                        for gamma in gamma_values:
                            try:
                                model = ExponentialSmoothing(
                                    data,
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=seasonal_periods
                                ).fit(
                                    smoothing_level=alpha,
                                    smoothing_trend=beta,
                                    smoothing_seasonal=gamma
                                )
                                fitted_values = model.fittedvalues
                                mse = mean_squared_error(data, fitted_values)
                                if mse < best_mse:
                                    best_mse = mse
                                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
                            except Exception:
                                pass  # Bỏ qua lỗi nếu có

                return best_alpha, best_beta, best_gamma, best_mse

            # Tập giá trị alpha, beta, gamma để thử nghiệm
            alpha_range = np.linspace(0.01, 1.0, 10)
            beta_range = np.linspace(0.01, 1.0, 10)
            gamma_range = np.linspace(0.01, 1.0, 10)

            # Nhập giá trị seasonal_periods từ người dùng
            seasonal_periods = st.number_input("Chu kỳ mùa vụ (Seasonal Periods)", min_value=1, value=12, step=1)

            # Tìm alpha, beta, gamma tối ưu
            with st.spinner("Đang tối ưu hóa các tham số..."):
                optimal_alpha, optimal_beta, optimal_gamma, optimal_mse = optimize_holt_winters(
                    stock_data['Close'], alpha_range, beta_range, gamma_range, seasonal_periods
                )

            # Áp dụng mô hình với alpha, beta, gamma tối ưu
            final_model = ExponentialSmoothing(
                stock_data['Close'],
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit(
                smoothing_level=optimal_alpha,
                smoothing_trend=optimal_beta,
                smoothing_seasonal=optimal_gamma
            )
            Holt_Winters_opt = final_model.fittedvalues

            # Dự báo giá tương lai từ end_date
            forecast_values = final_model.forecast(forecast_steps)
            future_dates = [stock_data.index[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]

            # Làm sạch dữ liệu dự báo
            future_dates_cleaned = [date.strftime('%Y-%m-%d') for date in future_dates]
            forecast_values_cleaned = [float(value) for value in forecast_values]

            # Chuẩn bị dữ liệu cho Altair
            holt_winters_df = pd.DataFrame({
                'Date': stock_data.index.to_list(),
                'Close': stock_data['Close'].values.flatten(),
                'Holt_Winters': Holt_Winters_opt.values.flatten()
            }).reset_index(drop=True)

            # Biểu đồ Altair với chú thích (legend)
            holt_winters_df_melted = pd.DataFrame({
                'Date': holt_winters_df['Date'].tolist() + holt_winters_df['Date'].tolist(),
                'Value': holt_winters_df['Close'].tolist() + holt_winters_df['Holt_Winters'].tolist(),
                'Type': ['Giá thực tế'] * len(holt_winters_df) +
                        [f'Holt-Winters (α={optimal_alpha:.2f}, β={optimal_beta:.2f}, γ={optimal_gamma:.2f})'] * len(holt_winters_df)
            })

            chart_holt_winters = alt.Chart(holt_winters_df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Ngày'),
                y=alt.Y('Value:Q', title='Giá cổ phiếu', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', title='Loại Dữ Liệu'),
                tooltip=[
                    alt.Tooltip('Date:T', title='Ngày'),
                    alt.Tooltip('Value:Q', title='Giá cổ phiếu', format=',.2f'),
                    alt.Tooltip('Type:N', title='Loại Dữ Liệu')
                ]
            ).properties(
                width=800, height=400,
                title=f"Biểu đồ Dự báo với Holt-Winters (α={optimal_alpha:.2f}, β={optimal_beta:.2f}, γ={optimal_gamma:.2f})"
            )

            st.altair_chart(chart_holt_winters, use_container_width=True)

            # Tạo bảng dữ liệu dự đoán
            forecast_df = pd.DataFrame({
                'Ngày': future_dates_cleaned,
                'Giá Dự Đoán': forecast_values_cleaned
            })

            st.write("### Bảng dữ liệu dự đoán Holt-Winters")
            st.dataframe(forecast_df[['Ngày', 'Giá Dự Đoán']])

            # Tính toán các chỉ số lỗi
            y_true = stock_data['Close'].values.flatten()
            Holt_Winters_opt = Holt_Winters_opt.reindex(stock_data.index, method='nearest').values.flatten()
            epsilon = 1e-10
            mae_holt_winter_opt = mean_absolute_error(y_true, Holt_Winters_opt)
            mse_holt_winter_opt = mean_squared_error(y_true, Holt_Winters_opt)
            rmse_holt_winter_opt = np.sqrt(mse_holt_winter_opt)
            mape_holt_winter_opt = np.mean(np.abs((y_true - Holt_Winters_opt) / (y_true + epsilon))) * 100

            # Hiển thị các chỉ số lỗi
            st.write("**Đánh giá hiệu suất:**")
            st.metric("Optimal Alpha", f"{optimal_alpha:.2f}")
            st.metric("Optimal Beta", f"{optimal_beta:.2f}")
            st.metric("Optimal Gamma", f"{optimal_gamma:.2f}")
            st.metric("MAE", f"{mae_holt_winter_opt:.2f}")
            st.metric("MSE", f"{mse_holt_winter_opt:.2f}")
            st.metric("RMSE", f"{rmse_holt_winter_opt:.2f}")
            st.metric("MAPE", f"{mape_holt_winter_opt:.2f}%")
