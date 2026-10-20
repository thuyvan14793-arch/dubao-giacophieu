import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(
    page_title="Stock Forecast AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 STOCK FORECAST AI")
st.caption(
    "Ứng dụng phân tích và dự báo giá cổ phiếu "
    "10 phiên giao dịch tiếp theo bằng Deep Learning (LSTM)."
)

st.info(
    "Mô hình sử dụng dữ liệu giá đóng cửa trong quá khứ. "
    "Kết quả chỉ mang tính tham khảo, không đảm bảo lợi nhuận đầu tư."
)

# =========================
# 2. THANH ĐIỀU KHIỂN
# =========================
with st.sidebar:
    st.header("⚙️ Thiết lập dự báo")

    ticker_input = st.text_input(
        "Mã cổ phiếu",
        value="VNM"
    )

    start_date = st.date_input(
        "Ngày bắt đầu",
        value=datetime.date(2018, 1, 1)
    )

    end_date = st.date_input(
        "Ngày kết thúc",
        value=datetime.date.today()
    )

    forecast_days = st.slider(
        "Số phiên cần dự báo",
        min_value=1,
        max_value=20,
        value=10
    )

    window_size = st.selectbox(
        "Số phiên dữ liệu đầu vào",
        options=[30, 60, 90],
        index=1
    )

    epochs = st.slider(
        "Số vòng huấn luyện tối đa",
        min_value=5,
        max_value=50,
        value=20
    )

    run_button = st.button(
        "🚀 Phân tích cổ phiếu",
        type="primary",
        use_container_width=True
    )

# Chuẩn hóa mã cổ phiếu
ticker = ticker_input.strip().upper()

if len(ticker) == 3 and "." not in ticker:
    ticker += ".VN"


# =========================
# 3. HÀM CHUẨN BỊ DỮ LIỆU
# =========================
def load_stock_data(symbol, start, end):
    df = yf.download(
        symbol,
        start=start,
        end=end + datetime.timedelta(days=1),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xử lý cấu trúc cột MultiIndex của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df["Close"]

        if isinstance(close_data, pd.DataFrame):
            if symbol in close_data.columns:
                close_data = close_data[symbol]
            else:
                close_data = close_data.iloc[:, 0]
    else:
        close_data = df["Close"]

    result = pd.DataFrame({
        "Close": pd.to_numeric(close_data, errors="coerce")
    }, index=df.index)

    result = result.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    result = result[result["Close"] > 0]
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]

    return result


def make_sequences(values, window):
    X, y = [], []

    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])

    return (
        np.asarray(X, dtype=np.float32).reshape(-1, window, 1),
        np.asarray(y, dtype=np.float32).reshape(-1, 1)
    )


# =========================
# 4. HUẤN LUYỆN VÀ DỰ BÁO
# =========================
if run_button:

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    try:
        with st.spinner(
            f"Đang tải dữ liệu {ticker} và huấn luyện mô hình..."
        ):

            # Tải dữ liệu
            df = load_stock_data(
                ticker,
                start_date,
                end_date
            )

            if len(df) <= window_size + 30:
                st.error(
                    "Dữ liệu không đủ để huấn luyện mô hình. "
                    "Hãy chọn khoảng thời gian dài hơn "
                    "hoặc giảm số phiên đầu vào."
                )
                st.stop()

            prices = df[["Close"]].values
            dates = df.index

            # Chia train/test theo thứ tự thời gian
            split_index = int(len(prices) * 0.8)

            train_prices = prices[:split_index]
            test_prices = prices[split_index:]

            if len(train_prices) <= window_size + 1:
                st.error("Tập huấn luyện không đủ dữ liệu.")
                st.stop()

            # Chuẩn hóa chỉ dựa trên tập train
            scaler = MinMaxScaler()

            train_scaled = scaler.fit_transform(train_prices)
            test_scaled = scaler.transform(test_prices)

            # Tạo chuỗi train
            X_train_all, y_train_all = make_sequences(
                train_scaled,
                window_size
            )

            if len(X_train_all) < 2:
                st.error("Không đủ mẫu để huấn luyện mô hình.")
                st.stop()

            # Tách validation theo thời gian
            val_size = max(
                1,
                int(len(X_train_all) * 0.1)
            )

            if val_size >= len(X_train_all):
                val_size = 1

            X_train = X_train_all[:-val_size]
            y_train = y_train_all[:-val_size]

            X_val = X_train_all[-val_size:]
            y_val = y_train_all[-val_size:]

            # Tạo tập test, sử dụng dữ liệu lịch sử
            # ngay trước thời điểm bắt đầu test
            test_combined = np.concatenate(
                [train_scaled[-window_size:], test_scaled],
                axis=0
            )

            X_test, y_test = make_sequences(
                test_combined,
                window_size
            )

            # Xây dựng mô hình LSTM
            model = Sequential([
                tf.keras.Input(
                    shape=(window_size, 1)
                ),
                LSTM(
                    64,
                    return_sequences=True
                ),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stopping],
                verbose=0
            )

            # =========================
            # 5. ĐÁNH GIÁ MÔ HÌNH
            # =========================
            test_pred_scaled = model.predict(
                X_test,
                verbose=0
            )

            test_pred = scaler.inverse_transform(
                test_pred_scaled
            ).flatten()

            y_test_real = scaler.inverse_transform(
                y_test
            ).flatten()

            rmse = np.sqrt(
                mean_squared_error(y_test_real, test_pred)
            )

            mae = mean_absolute_error(
                y_test_real,
                test_pred
            )

            r2 = (
                r2_score(y_test_real, test_pred)
                if len(y_test_real) >= 2
                else np.nan
            )

            # =========================
            # 6. DỰ BÁO TƯƠNG LAI
            # =========================
            # Dùng toàn bộ dữ liệu đã có để tạo cửa sổ
            # dự báo sau khi mô hình đã được đánh giá.
            all_scaled = scaler.transform(prices)

            last_window = all_scaled[
                -window_size:
            ].reshape(1, window_size, 1)

            future_scaled = []

            for _ in range(forecast_days):
                next_value = model.predict(
                    last_window,
                    verbose=0
                )[0, 0]

                future_scaled.append(next_value)

                last_window = np.concatenate(
                    [
                        last_window[:, 1:, :],
                        np.array(
                            [[[next_value]]],
                            dtype=np.float32
                        )
                    ],
                    axis=1
                )

            future_prices = scaler.inverse_transform(
                np.array(future_scaled).reshape(-1, 1)
            ).flatten()

            # Ngày làm việc dự kiến, không bao gồm cuối tuần
            future_dates = pd.bdate_range(
                start=dates[-1] + pd.Timedelta(days=1),
                periods=forecast_days
            )

            # =========================
            # 7. HIỂN THỊ KẾT QUẢ
            # =========================
            st.success(
                f"Đã tải {len(df):,} phiên giao dịch "
                f"của mã {ticker}."
            )

            st.subheader("📌 Tổng quan cổ phiếu")

            latest_price = float(prices[-1, 0])
            predicted_price = float(future_prices[0])

            change_pct = (
                (predicted_price - latest_price)
                / latest_price * 100
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Giá đóng cửa gần nhất",
                f"{latest_price:,.2f}"
            )

            col2.metric(
                f"Dự báo phiên kế tiếp",
                f"{predicted_price:,.2f}",
                f"{change_pct:+.2f}%"
            )

            col3.metric(
                "Số phiên dữ liệu",
                f"{len(df):,}"
            )

            # Chỉ số đánh giá
            st.divider()
            st.subheader("📊 Đánh giá mô hình trên tập Test")

            c1, c2, c3 = st.columns(3)

            c1.metric("RMSE", f"{rmse:,.4f}")
            c2.metric("MAE", f"{mae:,.4f}")
            c3.metric(
                "R² Score",
                f"{r2:.4f}" if np.isfinite(r2) else "N/A"
            )

            st.caption(
                "RMSE và MAE đo mức sai số dự báo. "
                "R² phản ánh mức độ phù hợp trên tập kiểm tra; "
                "giá trị âm vẫn có thể xảy ra."
            )

            # Bảng dự báo
            st.divider()
            st.subheader(
                f"🔮 Dự báo {forecast_days} phiên tiếp theo"
            )

            df_future = pd.DataFrame({
                "Ngày dự kiến": future_dates.strftime(
                    "%d/%m/%Y"
                ),
                "Giá dự báo": future_prices
            })

            st.dataframe(
                df_future.style.format({
                    "Giá dự báo": "{:,.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Cho phép tải bảng dự báo
            csv_data = df_future.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="📥 Tải bảng dự báo CSV",
                data=csv_data,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv"
            )

            # =========================
            # 8. BIỂU ĐỒ GIÁ
            # =========================
            st.divider()
            st.subheader("📈 Biểu đồ giá thực tế và dự báo")

            fig = go.Figure()

            # Giá lịch sử
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices.flatten(),
                mode="lines",
                name="Giá đóng cửa thực tế",
                line=dict(width=2)
            ))

            # Dự báo trên tập test
            test_dates = dates[split_index:]

            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_pred,
                mode="lines",
                name="Dự báo tập Test",
                line=dict(
                    width=1.5,
                    dash="dash"
                )
            ))

            # Dự báo tương lai
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode="lines+markers",
                name="Dự báo tương lai",
                line=dict(width=3),
                marker=dict(size=7)
            ))

            fig.update_layout(
                title=f"Phân tích giá cổ phiếu {ticker}",
                xaxis_title="Ngày giao dịch",
                yaxis_title="Giá đóng cửa",
                hovermode="x unified",
                template="plotly_white",
                height=550,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =========================
            # 9. BIỂU ĐỒ QUÁ TRÌNH HUẤN LUYỆN
            # =========================
            st.divider()
            st.subheader("🧠 Quá trình huấn luyện LSTM")

            loss_fig = go.Figure()

            loss_fig.add_trace(go.Scatter(
                y=history.history["loss"],
                mode="lines",
                name="Training Loss"
            ))

            loss_fig.add_trace(go.Scatter(
                y=history.history["val_loss"],
                mode="lines",
                name="Validation Loss"
            ))

            loss_fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Mean Squared Error",
                template="plotly_white",
                height=350
            )

            st.plotly_chart(
                loss_fig,
                use_container_width=True
            )

            st.caption(
                "Lưu ý: Ngày dự báo được tạo theo lịch ngày làm việc "
                "thông thường, chưa loại trừ các ngày nghỉ lễ của "
                "từng thị trường. Đây là dự báo tham khảo, "
                "không phải khuyến nghị mua hoặc bán cổ phiếu."
            )

    except Exception as e:
        st.error(
            f"Đã xảy ra lỗi khi phân tích dữ liệu: {e}"
        )
        st.info(
            "Hãy kiểm tra mã cổ phiếu, kết nối Internet, "
            "khoảng thời gian dữ liệu và các thư viện đã cài đặt."
        )

else:
    st.markdown("### 👋 Bắt đầu phân tích")
    st.write(
        "Nhập mã cổ phiếu ở thanh bên trái, chọn thời gian "
        "và nhấn **Phân tích cổ phiếu** để xem kết quả."
    )
