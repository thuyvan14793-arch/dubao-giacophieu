rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred) if len(y_test_real) > 1 else np.nan

print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print(f"RMSE: {rmse:,.4f}")
print(f"MAE : {mae:,.4f}")
print(f"R²  : {r2:.4f}")

# -------------------------
# 6. Biểu đồ loss và kết quả test
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f"LSTM - Train Loss vs Validation Loss ({TICKER})")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

test_dates = data_df.index[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_real, label="Giá thực tế")
plt.plot(test_dates, test_pred, label="Giá dự báo")
plt.title(f"Giá đóng cửa thực tế và dự báo trên tập test - {TICKER}")
plt.xlabel("Ngày")
plt.ylabel("Giá đóng cửa")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# 7. Dự báo các phiên tiếp theo
# -------------------------
last_window = all_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
future_scaled = []

for _ in range(FORECAST_DAYS):
    next_value = model.predict(last_window, verbose=0)[0, 0]
    future_scaled.append(next_value)
    next_step = np.array(next_value, dtype="float32").reshape(1, 1, 1)
    last_window = np.concatenate([last_window[:, 1:, :], next_step], axis=1)

future_prices = scaler.inverse_transform(
    np.array(future_scaled).reshape(-1, 1)
).ravel()

# Tạo ngày làm việc dự kiến; không bao gồm cuối tuần.
# Lịch này chưa loại trừ ngày nghỉ lễ của thị trường Việt Nam.
last_date = data_df.index[-1]
future_dates = pd.bdate_range(
    start=last_date + pd.offsets.BDay(1),
    periods=FORECAST_DAYS
)

forecast_df = pd.DataFrame({
    "Ngày dự kiến": future_dates.date,
    "Giá dự báo": future_prices
})

print(f"\n===== DỰ BÁO {FORECAST_DAYS} PHIÊN TIẾP THEO: {TICKER} =====")
print(forecast_df.to_string(index=False, formatters={
    "Giá dự báo": lambda x: f"{x:,.2f}"
}))

# -------------------------
# 8. Biểu đồ dự báo tương lai
# -------------------------
recent_n = min(90, len(data_df))
plt.figure(figsize=(12, 6))
plt.plot(
    data_df.index[-recent_n:],
    data_df["Close"].iloc[-recent_n:],
    label="Giá đóng cửa lịch sử"
)
plt.plot(
    future_dates,
    future_prices,
    marker="o",
    linestyle="--",
    label=f"Dự báo {FORECAST_DAYS} phiên"
)
plt.title(f"Dự báo giá cổ phiếu {TICKER} bằng LSTM")
plt.xlabel("Ngày")
plt.ylabel("Giá đóng cửa")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# 9. Lưu kết quả
# -------------------------
forecast_df.to_csv(
    f"du_bao_{TICKER.replace('.', '_')}.csv",
    index=False,
    encoding="utf-8-sig"
)
print(f"\nĐã lưu kết quả vào du_bao_{TICKER.replace('.', '_')}.csv")

print(
    "\nLưu ý: Đây là dự báo mô hình dựa trên giá lịch sử, "
    "không phải cam kết giá tương lai hay khuyến nghị mua/bán. "
    "Ngày dự kiến là ngày làm việc và chưa loại trừ ngày nghỉ lễ."
)

