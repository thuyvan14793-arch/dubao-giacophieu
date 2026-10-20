xaxis_title="Thời gian / Chỉ số phiên",
                    yaxis_title="Giá cổ phiếu",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'], mode='lines+markers', name='Train Loss',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'], mode='lines+markers', name='Validation Loss',
                    line=dict(color='#d62728', width=2)
                ))
                fig_loss.update_layout(
                    title="Đường cong Hàm Mất Mát qua các Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Loss (Mean Squared Error)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            # KHỐI 4: CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH
            st.markdown("---")
            st.subheader("📐 Chỉ Số Đánh Giá Mô Hình (Model Metrics)")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.4f}")
            c2.metric("MAE (Mean Absolute Error)", f"{mae:,.4f}")
            c3.metric("R² Score", f"{r2:.4f}")
