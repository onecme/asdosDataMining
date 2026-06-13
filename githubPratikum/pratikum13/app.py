import streamlit as st
import torch
import jcopdl  # noqa: F401  (diperlukan untuk unpickle configs.pth)
from torch import nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ====== Model Definition (harus sama dengan saat training) ======
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


# ====== Load Model & Artifacts ======
@st.cache_resource
def load_model():
    # weights_only=False diperlukan karena config disimpan sebagai
    # objek jcopdl.callback._config.Config (bukan dict biasa)
    config = torch.load(os.path.join(BASE_DIR, "configs.pth"), map_location="cpu", weights_only=False)

    model = LSTM(
        input_size=config.input_size,
        output_size=config.output_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "weights_best.pth"), map_location="cpu"))
    model.eval()

    return model, config


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


@st.cache_data
def load_last_window():
    df = pd.read_csv(os.path.join(BASE_DIR, "last_window.csv"), index_col=0, parse_dates=True)
    return df


model, config = load_model()
scaler = load_scaler()
last_window_df = load_last_window()

seq_len = config.seq_len


# ====== Forecasting Function ======
def forecast(model, last_window_df, scaler, seq_len, n_future):
    # ambil kolom Close & normalisasi
    values = last_window_df["Close"].values.reshape(-1, 1)
    values_scaled = scaler.transform(values)

    window = torch.tensor(values_scaled[-seq_len:], dtype=torch.float32)
    window = window.unsqueeze(0)  # shape: (1, seq_len, 1)

    future_preds = []
    with torch.no_grad():
        for _ in range(n_future):
            output, _ = model(window, None)
            next_val = output[:, -1:, :]
            future_preds.append(next_val.item())

            window = torch.cat([window[:, 1:, :], next_val], dim=1)

    future_preds_real = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

    return future_preds_real


# ====== Analisis dengan Groq ======
def generate_gemini_analysis(hist_real, future_preds_real, n_future, last_date, future_dates):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return "⚠️ GROQ_API_KEY belum diset di Streamlit secrets."

    client = Groq(api_key=api_key)

    perubahan_persen = (future_preds_real[-1] - hist_real[-1]) / hist_real[-1] * 100
    arah = "naik" if perubahan_persen > 0 else "turun"

    data_text = "\n".join(
        f"{d.strftime('%Y-%m-%d')}: {p:.2f}"
        for d, p in zip(future_dates, future_preds_real)
    )

    prompt = f"""
Kamu adalah analis keuangan. Berikut hasil forecasting harga saham menggunakan model LSTM.

Harga terakhir (aktual) pada {last_date.strftime('%Y-%m-%d')}: {hist_real[-1]:.2f}

Hasil prediksi {n_future} hari ke depan:
{data_text}

Secara keseluruhan, harga diprediksi {arah} sekitar {abs(perubahan_persen):.2f}% dari harga terakhir.

Tulis analisis singkat (3-5 kalimat) dalam Bahasa Indonesia mengenai tren hasil prediksi ini.
Sertakan: (1) interpretasi arah tren, (2) catatan bahwa ini hasil model statistik dan bukan saran investasi,
(3) saran umum untuk tetap memantau data terbaru.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gagal mendapatkan analisis dari Groq: {e}"


# ====== Streamlit UI ======
st.title("📈 Prediksi Harga Saham dengan LSTM")
st.write("Forecasting harga Close saham berdasarkan model LSTM yang telah dilatih.")

n_future = st.slider("Pilih jumlah hari prediksi ke depan:", min_value=1, max_value=7, value=3)

if st.button("Prediksi"):
    with st.spinner("Sedang menghitung prediksi..."):
        future_preds_real = forecast(model, last_window_df, scaler, seq_len, n_future)

    # buat tanggal untuk hasil forecast
    last_date = last_window_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future)

    result_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi Harga": future_preds_real
    })

    st.subheader("Hasil Prediksi")
    st.dataframe(result_df, use_container_width=True)

    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 5))

    # data historis (asli)
    hist_real = last_window_df["Close"].values
    ax.plot(range(len(hist_real)), hist_real, label="Data Historis (14 hari terakhir)")

    # forecast
    ax.plot(
        range(len(hist_real) - 1, len(hist_real) - 1 + n_future + 1),
        np.concatenate([[hist_real[-1]], future_preds_real]),
        '--', label=f"Forecast {n_future} Hari"
    )

    ax.set_title(f"Forecast {n_future} Hari ke Depan")
    ax.set_xlabel("Hari")
    ax.set_ylabel("Harga Close")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Analisis Gemini
    st.subheader("🤖 Analisis AI (Gemini)")
    with st.spinner("Sedang membuat analisis..."):
        analysis = generate_gemini_analysis(
            hist_real, future_preds_real, n_future, last_date, future_dates
        )
    st.write(analysis)