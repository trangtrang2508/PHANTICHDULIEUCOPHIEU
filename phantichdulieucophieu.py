import streamlit as st  
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import warnings
import plotly.graph_objects as go
import altair as alt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
warnings.filterwarnings("ignore")

# Cài đặt Streamlit
st.set_page_config(page_title="Phân tích dữ liệu cổ phiếu", layout="wide", initial_sidebar_state="expanded")
st.title("Phân tích dữ liệu cổ phiếu")

# Khởi tạo danh sách mã cổ phiếu và tên công ty
tickers = ['MSFT', 'ORCL', 'IBM', 'CRM', 'SAP']
company_names = {'MSFT': 'Microsoft', 'ORCL': 'Oracle', 'IBM': 'IBM', 'CRM': 'Salesforce', 'SAP': 'SAP'}

# Sidebar: Bộ lọc
st.sidebar.title("Bộ lọc")
selected_ticker = st.sidebar.selectbox("Chọn công ty:", tickers)
selected_company = company_names[selected_ticker]
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", datetime.now())

# Tải dữ liệu cổ phiếu
stock_data = yf.download(selected_ticker, start=start_date, end=end_date)

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
    tab1, tab2, tab3 = st.tabs(["Thống kê mô tả", "Chuỗi thời gian", "Mô hình dự báo"])

    # ----------- Tab 1: Thống kê mô tả ----------- #
    with tab1:
        # Biểu đồ giá đóng theo ngày
        if st.checkbox("Biểu đồ giá đóng theo ngày"):
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
            st.caption("Biểu đồ này thể hiện giá đóng của cổ phiếu theo từng ngày, giúp theo dõi xu hướng biến động của giá.")

        # Trung bình giá đóng cửa mỗi năm
        if st.checkbox("Trung bình giá đóng cửa mỗi năm"):
            avg_close_by_year = stock_data.groupby('Year')['Close'].mean().reset_index()
            st.write("Trung bình giá đóng cửa mỗi năm:")
            st.dataframe(avg_close_by_year)

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
            st.caption("Biểu đồ này hiển thị giá trung bình đóng cửa của cổ phiếu theo từng năm.")

        # Trung bình giá đóng cửa theo quý
        if st.checkbox("Trung bình giá đóng cửa theo quý"):
            avg_close_by_quarter = stock_data.groupby(['Year', 'Quarter'])['Close'].mean().reset_index()
            quarter_chart = alt.Chart(avg_close_by_quarter).mark_bar().encode(
                x=alt.X('Year:O', title='Năm'),
                y=alt.Y('Close:Q', title='Giá trung bình (USD)'),
                color=alt.Color('Quarter:O', title='Quý'),
                tooltip=[alt.Tooltip('Year:O', title='Năm'),
                         alt.Tooltip('Quarter:O', title='Quý'),
                         alt.Tooltip('Close:Q', title='Giá trung bình', format=',.2f')]
            ).properties(
                title="Trung bình giá đóng cửa theo quý từ 2020 đến 2024",
                width=800,
                height=400
            )
            st.altair_chart(quarter_chart, use_container_width=True)
            st.caption("Biểu đồ này thể hiện giá trung bình đóng cửa của cổ phiếu theo từng quý trong các năm.")

        # Biểu đồ khối lượng giao dịch
        if st.checkbox("Khối lượng cổ phiếu giao dịch theo ngày"):
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
            st.caption("Biểu đồ này giúp theo dõi khối lượng giao dịch cổ phiếu theo từng ngày.")

        # Tổng lượng cổ phiếu giao dịch mỗi năm
        if st.checkbox("Tổng lượng cổ phiếu giao dịch mỗi năm"):
            volume_by_year = stock_data.groupby('Year')['Volume'].sum().reset_index()
            volume_by_year_chart = alt.Chart(volume_by_year).mark_bar(color='green').encode(
                x=alt.X('Year:O', title='Năm'),
                y=alt.Y('Volume:Q', title='Tổng lượng giao dịch', axis=alt.Axis(format=',.0f')),
                tooltip=[alt.Tooltip('Year:O', title='Năm'),
                         alt.Tooltip('Volume:Q', title='Tổng lượng giao dịch', format=',.0f')]
            ).properties(
                title="Tổng lượng giao dịch cổ phiếu từ năm 2020 đến 2024",
                width=800,
                height=400
            )
            st.altair_chart(volume_by_year_chart, use_container_width=True)
            st.caption("Biểu đồ này thể hiện tổng khối lượng giao dịch cổ phiếu mỗi năm.")

        # Tổng lượng cổ phiếu giao dịch theo quý
        if st.checkbox("Tổng lượng cổ phiếu giao dịch theo quý"):
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
            st.caption("Biểu đồ này hiển thị tổng khối lượng giao dịch cổ phiếu theo từng quý.")

        # Biểu đồ tương quan các biến
        if st.checkbox("Biểu đồ tương quan các biến"):
            corr_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr().reset_index()
            corr_data = corr_data.melt(id_vars='index', var_name='Variable 2', value_name='Correlation')
            corr_data.rename(columns={'index': 'Variable 1'}, inplace=True)

            corr_chart = alt.Chart(corr_data).mark_rect().encode(
                x=alt.X('Variable 2:N', title='Biến 2', sort=None),
                y=alt.Y('Variable 1:N', title='Biến 1', sort=None),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blues'), title='Tương quan'),
                tooltip=[alt.Tooltip('Variable 1:N', title='Biến 1'),
                         alt.Tooltip('Variable 2:N', title='Biến 2'),
                         alt.Tooltip('Correlation:Q', title='Hệ số tương quan', format='.2f')]
            ).properties(
                title="Biểu đồ heatmap tương quan các biến",
                width=600,
                height=400
            ).interactive()  # Make chart interactive

            st.altair_chart(corr_chart, use_container_width=True)
            st.caption("Biểu đồ heatmap này thể hiện hệ số tương quan giữa các biến: Mở, Cao, Thấp, Đóng, Điều chỉnh và Khối lượng.")

        # Biểu đồ tỷ suất lợi nhuận hàng ngày (histplot với tần suất)
        if st.checkbox("Tỷ suất lợi nhuận hàng ngày"):
            # Tính tỷ suất lợi nhuận hàng ngày
            stock_data['Daily Returns'] = stock_data['Close'].pct_change()

            # Chỉ lấy dữ liệu từ năm 2024
            daily_returns_2024 = stock_data['Daily Returns']['2024-01-01':'2024-11-30']

            if not daily_returns_2024.empty:
                # Chuẩn bị dữ liệu cho biểu đồ
                daily_returns_df = daily_returns_2024.dropna().reset_index()

                # Tạo biểu đồ histplot cho tỷ suất lợi nhuận hàng ngày với tần suất
                daily_returns_histplot = alt.Chart(daily_returns_df).mark_bar().encode(
                    x=alt.X('Daily Returns:Q', bin=alt.Bin(maxbins=50), title='Tỷ suất lợi nhuận hàng ngày'),
                    y=alt.Y('count():Q', title='Tần suất'),
                    tooltip=[alt.Tooltip('Daily Returns:Q', title='Tỷ suất lợi nhuận', format='.4f'),
                             alt.Tooltip('count():Q', title='Tần suất', format='.4f')]
                ).transform_density(
                    'Daily Returns',  # Cột dữ liệu cần tính tần suất
                    as_=['Daily Returns', 'Density'],  # Các tên cột kết quả
                    bandwidth=0.001  # Độ mượt của tần suất
                ).mark_area().encode(
                    y='Density:Q'  # Dùng tần suất ở trục y
                ).properties(
                    title=f"Biểu đồ tỷ suất lợi nhuận hàng ngày của {selected_company} (2024)",
                    width=800,
                    height=400
                ).interactive()  # Tạo biểu đồ có thể tương tác

                # Hiển thị biểu đồ
                st.altair_chart(daily_returns_histplot, use_container_width=True)
                st.caption("Biểu đồ này hiển thị tỷ suất lợi nhuận hàng ngày của cổ phiếu trong năm 2024.")

            else:
                st.warning("Không có dữ liệu tỷ suất lợi nhuận hàng ngày cho năm 2024.")

        # Biểu đồ hình nến
        if st.checkbox("Biểu đồ hình nến"):
            st.subheader(f"Biểu đồ hình nến của {selected_company}")

            # Đảm bảo start_date và end_date là kiểu datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Lọc dữ liệu trong khoảng thời gian người dùng chọn
            candlestick_data = stock_data[['Open', 'High', 'Low', 'Close']].copy()

            # Lọc dữ liệu theo khoảng thời gian
            filtered_data = candlestick_data[(candlestick_data.index >= start_date) & (candlestick_data.index <= end_date)]

            # Kiểm tra xem có dữ liệu sau khi lọc không
            if filtered_data.empty:
                st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
            else:
                # Tạo biểu đồ nến
                candlestick_chart = alt.Chart(filtered_data.reset_index()).mark_bar().encode(
                    x=alt.X('Date:T', title='Ngày'),
                    y=alt.Y('Close:Q', title='Giá đóng (USD)'),
                    color=alt.condition(
                        alt.datum.Close > alt.datum.Open, 
                        alt.value('green'), alt.value('red')
                    ),
                    tooltip=[
                        alt.Tooltip('Date:T', title='Ngày', format='%Y-%m-%d'),
                        alt.Tooltip('Open:Q', title='Giá mở cửa', format=',.2f'),
                        alt.Tooltip('High:Q', title='Giá cao nhất', format=',.2f'),
                        alt.Tooltip('Low:Q', title='Giá thấp nhất', format=',.2f'),
                        alt.Tooltip('Close:Q', title='Giá đóng cửa', format=',.2f')
                    ]
                ).properties(
                    title=f"Biểu đồ hình nến của {selected_company} từ {start_date.date()} đến {end_date.date()}",
                    width=800,
                    height=400
                ).interactive()

                # Hiển thị biểu đồ
                st.altair_chart(candlestick_chart, use_container_width=True)

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
        st.write("### Lựa chọn khung thời gian dự báo")
        forecast_type = st.radio(
            "Chọn khung thời gian dự báo:",
            ('Ngày', 'Tuần', 'Tháng'),
            horizontal=True
        )

        # Xử lý dữ liệu theo khung thời gian được chọn
        if forecast_type == 'Ngày':
            data_resampled = stock_data['Close']  # Dữ liệu giữ nguyên
            st.write("Dự báo dựa trên dữ liệu hằng ngày.")
        elif forecast_type == 'Tuần':
            data_resampled = stock_data['Close'].resample('W').mean()
            st.write("Dự báo dựa trên dữ liệu trung bình tuần.")
        elif forecast_type == 'Tháng':
            data_resampled = stock_data['Close'].resample('M').mean()
            st.write("Dự báo dựa trên dữ liệu trung bình tháng.")

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
            valid_indices = MA_values.dropna().index  # Chỉ lấy các chỉ số hợp lệ

            # Chuyển đổi dữ liệu thành mảng 1 chiều
            ma_df = pd.DataFrame({
                'Date': data_resampled.loc[valid_indices].index.to_list(),  # Đảm bảo là danh sách
                'Close': data_resampled.loc[valid_indices].values.flatten(),  # Chuyển thành mảng 1 chiều
                'Moving Average': MA_values.dropna().values.flatten()  # Chuyển thành mảng 1 chiều
            }).reset_index(drop=True)

            # Chuẩn bị dữ liệu để tạo legend
            ma_df_melted = pd.DataFrame({
                'Date': ma_df['Date'].tolist() + ma_df['Date'].tolist(),
                'Giá Trị': ma_df['Close'].tolist() + ma_df['Moving Average'].tolist(),
                'Loại Dữ Liệu': ['Actual Close Price'] * len(ma_df) + [f'Moving Average (Window = {window_size})'] * len(ma_df)
            })

            # Biểu đồ Altair với legend
            chart_with_legend = alt.Chart(ma_df_melted).mark_line().encode(
                x=alt.X('Date:T', title='Ngày'),
                y=alt.Y('Giá Trị:Q', title='Giá Cổ Phiếu', scale=alt.Scale(zero=False)),
                color=alt.Color('Loại Dữ Liệu:N', title='Legend'),  # Tạo legend từ cột 'Loại Dữ Liệu'
                tooltip=[
                    alt.Tooltip('Date:T', title='Ngày'),
                    alt.Tooltip('Giá Trị:Q', title='Giá Cổ Phiếu', format=',.2f'),
                    alt.Tooltip('Loại Dữ Liệu:N', title='Loại')
                ]
            ).properties(
                width=800,
                height=400,
                title=f"Biểu đồ Giá Cổ Phiếu và Moving Average ({ma_option})"
            )

            st.altair_chart(chart_with_legend, use_container_width=True)

            # Tính toán các chỉ số hiệu suất
            st.write("### Đánh giá hiệu suất dự đoán:")
            y_true = data_resampled.loc[valid_indices].values.flatten()
            y_pred = MA_values.dropna().values.flatten()

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)

            st.metric("MAE", f"{mae:.2f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAPE", f"{mape:.2f} %")
            st.metric("R²", f"{r2:.2f}")

        # Dự báo bằng Exponential Smoothing
        use_exp_smoothing = st.checkbox("Exponential Smoothing")
        if use_exp_smoothing:
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

            # Dự báo cho 60 ngày tiếp theo
            forecast_60_days = model.forecast(60)

            # Vẽ biểu đồ
            st.write(f"Biểu đồ Exponential Smoothing (alpha tối ưu = {optimal_alpha:.2f}):")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data['Close'], label="Giá thực tế", color="blue", alpha=0.8)
            ax.plot(SES_opt, label=f"Exponential Smoothing (alpha = {optimal_alpha:.2f})", color="orange", linewidth=2)
            ax.plot(pd.date_range(stock_data.index[-1], periods=61, freq='D')[1:], forecast_60_days, label="Dự báo 60 ngày", color="green", linestyle="--")
            ax.set_title("Exponential Smoothing (Hệ số tối ưu)", fontsize=16)
            ax.set_xlabel("Ngày", fontsize=14)
            ax.set_ylabel("Giá", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Tính toán các chỉ số lỗi
            mae_SES_opt = mean_absolute_error(stock_data['Close'], SES_opt)
            mse_SES_opt = mean_squared_error(stock_data['Close'], SES_opt)
            rmse_SES_opt = np.sqrt(mse_SES_opt)
        
            # Hiển thị các chỉ số lỗi
            st.write("**Đánh giá hiệu suất:**")
            st.metric("MAE", f"{mae_SES_opt:.2f}")
            st.metric("MSE", f"{mse_SES_opt:.2f}")
            st.metric("RMSE", f"{rmse_SES_opt:.2f}")

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

            # Dự báo cho 60 ngày tiếp theo
            forecast_60_days_holt = final_model.forecast(60)

            # Vẽ biểu đồ
            st.write(f"Biểu đồ dự báo với Holt (α={optimal_alpha:.2f}, β={optimal_beta:.2f}):")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data['Close'], label="Giá thực tế", color="blue", alpha=0.8)
            ax.plot(Holt_opt, label=f"Holt (α={optimal_alpha:.2f}, β={optimal_beta:.2f})", color="orange", linewidth=2)
            ax.plot(pd.date_range(stock_data.index[-1], periods=61, freq='D')[1:], forecast_60_days_holt, label="Dự báo 60 ngày", color="green", linestyle="--")
            ax.set_title("Holt Model with Optimal Alpha and Beta", fontsize=16)
            ax.set_xlabel("Ngày", fontsize=14)
            ax.set_ylabel("Giá", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Tính toán các chỉ số lỗi
            mae_holt_opt = mean_absolute_error(stock_data['Close'], Holt_opt)
            mse_holt_opt = mean_squared_error(stock_data['Close'], Holt_opt)
            rmse_holt_opt = np.sqrt(mse_holt_opt)
            mape_holt_opt = np.mean(np.abs((stock_data['Close'] - Holt_opt) / stock_data['Close'])) * 100

            # Hiển thị các chỉ số lỗi
            st.write("**Đánh giá hiệu suất:**")
            st.metric("Optimal Alpha", f"{optimal_alpha:.2f}")
            st.metric("Optimal Beta", f"{optimal_beta:.2f}")
            st.metric("MAE", f"{mae_holt_opt:.2f}")
            st.metric("MSE", f"{mse_holt_opt:.2f}")
            st.metric("RMSE", f"{rmse_holt_opt:.2f}")
            st.metric("MAPE", f"{mape_holt_opt:.2f}")

        # Dự báo bằng Holt Winters
        use_holt_winters = st.checkbox("Holt Winters")
        if use_holt_winters:
            # Hàm tìm alpha, beta, gamma tối ưu
            def optimize_holt_winters(data, alpha_values, beta_values, gamma_values, seasonal_periods):
                best_alpha = None
                best_beta = None
                best_gamma = None
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
                                    best_alpha = alpha
                                    best_beta = beta
                                    best_gamma = gamma
                            except Exception:
                                # Bỏ qua các trường hợp không hợp lệ
                                pass

                return best_alpha, best_beta, best_gamma, best_mse

            # Tập giá trị alpha, beta, gamma để thử nghiệm
            alpha_range = np.linspace(0.01, 1.0, 10)
            beta_range = np.linspace(0.01, 1.0, 10)
            gamma_range = np.linspace(0.01, 1.0, 10)

            # Tìm alpha, beta, gamma tối ưu
            seasonal_periods = st.number_input("Chu kỳ mùa vụ (Seasonal Periods)", min_value=1, value=12, step=1)
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

            # Dự báo cho 60 ngày tiếp theo
            forecast_60_days_hw = final_model_hw.forecast(60)

            # Vẽ biểu đồ
            st.write(f"Biểu đồ dự báo với Holt-Winters (α={optimal_alpha:.2f}, β={optimal_beta:.2f}, γ={optimal_gamma:.2f}):")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data['Close'], label="Giá thực tế", color="blue", alpha=0.8)
            ax.plot(Holt_Winters_opt, label=f"Holt-Winters (α={optimal_alpha:.2f}, β={optimal_beta:.2f}, γ={optimal_gamma:.2f})", color="orange", linewidth=2)
            ax.plot(pd.date_range(stock_data.index[-1], periods=61, freq='D')[1:], forecast_60_days_hw, label="Dự báo 60 ngày", color="green", linestyle="--")
            ax.set_title("Holt-Winters Model with Optimal Alpha, Beta, and Gamma", fontsize=16)
            ax.set_xlabel("Ngày", fontsize=14)
            ax.set_ylabel("Giá", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Tính toán các chỉ số lỗi
            mae_holt_winter_opt = mean_absolute_error(stock_data['Close'], Holt_Winters_opt)
            mse_holt_winter_opt = mean_squared_error(stock_data['Close'], Holt_Winters_opt)
            rmse_holt_winter_opt = np.sqrt(mse_holt_winter_opt)
            mape_holt_winter_opt = np.mean(np.abs((stock_data['Close'] - Holt_Winters_opt) / stock_data['Close'])) * 100

            # Hiển thị các chỉ số lỗi
            st.write("**Đánh giá hiệu suất:**")
            st.metric("Optimal Alpha", f"{optimal_alpha:.2f}")
            st.metric("Optimal Beta", f"{optimal_beta:.2f}")
            st.metric("Optimal Gamma", f"{optimal_gamma:.2f}")
            st.metric("MAE", f"{mae_holt_winter_opt:.2f}")
            st.metric("MSE", f"{mse_holt_winter_opt:.2f}")
            st.metric("RMSE", f"{rmse_holt_winter_opt:.2f}")
            st.metric("MAPE", f"{mape_holt_winter_opt:.2f}")