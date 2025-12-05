import streamlit as st
from streamlit_option_menu import option_menu  # masih boleh, walau belum dipakai
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================
# CONFIG APP
# ==========================
st.set_page_config(
    page_title="MachineGuard AI - Machine Failure Prediction",
    page_icon="🛡️",
    layout="wide",
)

# ==========================
# KONSTANTA
# ==========================
FINAL_THRESHOLD = 0.28  # threshold akhir dari tuning kamu
AUC_TEST = 0.9479
RECALL_TEST = 0.7541
PRECISION_TEST = 0.3407
F1_TEST = 0.4694

# path gambar utama
IMG_ROC_PATH = Path("images/roc_xgb_test.png")
IMG_CM_PATH = Path("images/cm_xgb_test.png")
IMG_CUM_GAIN = Path("images/cum_gain_xgb_test.png")
IMG_LIFT = Path("images/lift_xgb_test.png")

# SHAP images
IMG_SHAP_SUMMARY = Path("images/shap_summary.png")
IMG_SHAP_TORQUE = Path("images/shap_dependence/Torque_Nm.png")
IMG_SHAP_TOOL_WEAR = Path("images/shap_dependence/Tool_wear_min.png")
IMG_SHAP_AIR_TEMP = Path("images/shap_dependence/Air_temperature_K.png")

# ==========================
# FUNGSI UTILITAS
# ==========================

@st.cache_resource
def load_pipeline():
    """
    Load pipeline (preprocessing + model) yang sudah kamu simpan dari Colab.
    """
    model_path = Path("artifacts/model_pipeline.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_failure(input_data: dict):
    """
    input_data: dict dengan key sesuai nama kolom fitur mentah.
    Return: (label, proba)
    """
    pipeline = load_pipeline()
    if pipeline is None:
        return None, None

    df = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(df)[0, 1]
    label = int(proba >= FINAL_THRESHOLD)
    return label, proba


def risk_text_and_color(label: int, proba: float):
    """
    Mengubah label & proba jadi teks dan warna (untuk styling UI).
    """
    if label == 1:
        return (
            "High Failure Risk",
            f"Mesin ini diprediksi memiliki risiko FAILURE tinggi ({proba:.1%}). "
            f"Disarankan untuk diprioritaskan inspeksi/maintenance.",
            "🔴",
        )
    else:
        return (
            "Low Failure Risk",
            f"Mesin ini diprediksi memiliki risiko failure rendah ({proba:.1%}). "
            f"Meski begitu, tetap lakukan monitoring berkala.",
            "🟢",
        )


# ==========================
# LAYOUT: SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("MachineGuard AI")
st.sidebar.markdown("Sistem Prediksi **Machine Failure** berbasis XGBoost.")

# Inisialisasi halaman aktif
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

def go_to_predict():
    st.session_state.page = "Mulai Prediksi"

page = st.sidebar.radio(
    "Navigasi",
    [
        "Beranda",
        "Tentang Sistem",
        "Teknologi & Proses",
        "Mulai Prediksi",
        "Insight Model",
    ],
    key="page",
)

# ==========================
# HALAMAN: BERANDA
# ==========================
if page == "Beranda":
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 🛡️ MachineGuard AI")
        st.markdown(
            """
            # Sistem Prediksi Machine Failure Berbasis AI

            MachineGuard AI membantu engineer dan analis maintenance
            untuk **mengidentifikasi mesin dengan risiko failure tinggi**
            sebelum benar-benar terjadi kerusakan.

            Dengan memanfaatkan **data sensor proses** dan model
            **XGBoost yang sudah dioptimasi**, sistem ini memberikan:
            - Prediksi gagal / tidak gagal
            - Probabilitas risiko failure
            - Insight fitur yang berkontribusi terhadap failure
            """
        )

        st.button("🚀 Mulai Prediksi Sekarang", on_click=go_to_predict)

    with col_right:
        st.markdown("#### Ringkasan Performa Model (Test Set)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC-ROC", f"{AUC_TEST:.3f}")
            st.metric("Recall (Failure)", f"{RECALL_TEST:.2f}")
        with col2:
            st.metric("Precision (Failure)", f"{PRECISION_TEST:.2f}")
            st.metric("F1-Score", f"{F1_TEST:.2f}")

        st.markdown("---")
        st.markdown("#### Info Dataset")
        st.markdown(
            """
            - Jenis masalah: **Klasifikasi biner**
            - Target: `Machine_failure` (0 = normal, 1 = fail)
            - Karakteristik: **Imbalanced** (failure ~3–4% dari data)
            """
        )

# ==========================
# HALAMAN: TENTANG SISTEM
# ==========================
elif page == "Tentang Sistem":
    st.markdown("## ℹ️ Tentang MachineGuard AI")

    st.markdown(
        """
        MachineGuard AI dikembangkan untuk membantu perusahaan manufaktur
        dalam melakukan **predictive maintenance**.

        ### Tujuan Utama
        - Mengidentifikasi **mesin yang berpotensi mengalami failure**
          berdasarkan data operasi harian.
        - Menyediakan **probabilitas risiko** beserta **label fail / non-fail**.
        - Membantu engineer untuk **memprioritaskan** mesin mana yang perlu
          dicek terlebih dahulu.

        ### Manfaat Bisnis
        - Mengurangi **downtime tak terduga**.
        - Mengurangi kerugian akibat **kerusakan mendadak**.
        - Membantu penggunaan resource maintenance yang lebih **efisien**:
          fokus pada mesin dengan risiko tertinggi.
        """
    )

    st.markdown("---")
    st.markdown("### Fitur yang Digunakan")
    st.markdown(
        """
        Beberapa contoh fitur input yang digunakan model:
        - `Type` (jenis mesin / produk)
        - `Air_temperature_K`
        - `Process_temperature_K`
        - `Rotational_speed_rpm`
        - `Torque_Nm`
        - `Tool_wear_min`
        """
    )

# ==========================
# HALAMAN: TEKNOLOGI & PROSES
# ==========================
elif page == "Teknologi & Proses":
    st.markdown("## ⚙️ Teknologi & Proses Deteksi")

    st.markdown("### 1. Model yang Digunakan")
    st.markdown(
        """
        - Model utama: **XGBoost Classifier**
        - Penanganan data imbalanced:
          - `scale_pos_weight` pada XGBoost
          - Fokus metrik pada **AUC** dan **recall** kelas failure
        - Model lain yang sempat diuji sebagai pembanding:
          - Random Forest
          - Logistic Regression
          - LightGBM
          - CatBoost
        """
    )

    st.markdown("---")
    st.markdown("### 2. Proses Deteksi Machine Failure")

    cols = st.columns(4)

    with cols[0]:
        st.markdown("#### 1️⃣ Input Data")
        st.markdown(
            """
            - Data sensor produksi
            - Parameter proses
            - Informasi jenis mesin
            """
        )

    with cols[1]:
        st.markdown("#### 2️⃣ Preprocessing")
        st.markdown(
            """
            - Encoding fitur kategorikal (`Type`)
            - Standardisasi fitur numerik
            - Penanganan multikolinearitas
            """
        )

    with cols[2]:
        st.markdown("#### 3️⃣ Analisis Model")
        st.markdown(
            """
            - Model XGBoost mempelajari pola failure
            - Optimasi hyperparameter dengan **Optuna**
            - Threshold disesuaikan untuk menaikkan recall
            """
        )

    with cols[3]:
        st.markdown("#### 4️⃣ Hasil & Prioritas")
        st.markdown(
            """
            - Prediksi **fail / non-fail**
            - **Probabilitas risiko failure**
            - Mesin berisiko tinggi bisa diprioritaskan untuk inspeksi
            """
        )

    st.markdown("---")
    st.markdown("### 3. Ringkasan Threshold Tuning")
    st.markdown(
        f"""
        - Threshold default (0.5) ➜ recall cukup tinggi, tapi masih bisa ditingkatkan.
        - Dilakukan pencarian threshold pada data validasi dengan kriteria:
          - **Recall ≥ 0.80** untuk kelas failure
          - F1-score maksimum di antara kandidat threshold tersebut
        - Hasil:
          - Threshold terbaik: **{FINAL_THRESHOLD:.2f}**
          - Memberi trade-off:
            - Recall naik (lebih banyak failure yang tertangkap)
            - Precision turun (lebih banyak false positive) namun masih wajar
        """
    )

# ==========================
# HALAMAN: MULAI PREDIKSI
# ==========================
elif page == "Mulai Prediksi":
    st.markdown("## 🚀 Mulai Prediksi Machine Failure")

    st.markdown(
        """
        Isi parameter mesin di bawah ini untuk mendapatkan
        **prediksi risiko machine failure**.
        """
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            type_machine = st.selectbox(
                "Type Mesin / Produk",
                options=["L", "M", "H"],
                index=1,
                help="Isi sesuai kategori Type di dataset (mis. L / M / H).",
            )

            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=250.0,
                max_value=400.0,
                value=300.0,
                step=0.5,
            )

            process_temp = st.number_input(
                "Process Temperature [K]",
                min_value=250.0,
                max_value=500.0,
                value=310.0,
                step=0.5,
            )

        with col2:
            rotational_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=0.0,
                max_value=3000.0,
                value=1500.0,
                step=10.0,
            )

            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5,
            )

            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0.0,
                max_value=400.0,
                value=100.0,
                step=1.0,
            )

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_data = {
            "Type": type_machine,
            "Air_temperature_K": air_temp,
            "Process_temperature_K": process_temp,
            "Rotational_speed_rpm": rotational_speed,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear,
        }

        label, proba = predict_failure(input_data)

        if label is None:
            st.error(
                "Model belum ditemukan. Pastikan file "
                "`artifacts/model_pipeline.pkl` sudah ada."
            )
        else:
            st.markdown("### Hasil Prediksi")

            risk_label, risk_desc, emoji = risk_text_and_color(label, proba)

            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.markdown(f"#### {emoji} {risk_label}")
                st.markdown(risk_desc)

                st.markdown("##### Probabilitas Failure")
                st.write(f"**{proba:.2%}**")

                st.markdown("##### Threshold yang Digunakan")
                st.write(
                    f"Model menggunakan threshold **{FINAL_THRESHOLD:.2f}** "
                    f"untuk mengubah probabilitas menjadi label fail / non-fail."
                )

            with col_right:
                st.markdown("##### Visual Probabilitas")
                st.progress(float(np.clip(proba, 0.0, 1.0)))
                st.caption("Bar di atas menunjukkan seberapa besar probabilitas failure.")

            st.markdown("---")
            st.markdown("#### Input yang Kamu Masukkan")
            st.dataframe(pd.DataFrame([input_data]))

# ==========================
# HALAMAN: INSIGHT MODEL
# ==========================
elif page == "Insight Model":
    st.markdown("## 📊 Insight Model")

    # 1. Metrik
    st.markdown("### 1. Performa Model di Test Set")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{AUC_TEST:.3f}")
    col2.metric("Recall (Failure)", f"{RECALL_TEST:.2f}")
    col3.metric("Precision (Failure)", f"{PRECISION_TEST:.2f}")
    col4.metric("F1-Score", f"{F1_TEST:.2f}")

    st.markdown("---")
    st.markdown("### 2. Interpretasi Model (Global)")

    st.markdown(
        """
        Dari analisis SHAP di notebook:

        - Fitur seperti **`Tool_wear_min`**, **`Torque_Nm`**, dan kombinasi
          kondisi proses (temperatur & kecepatan putar) memiliki kontribusi besar
          dalam memprediksi failure.
        - Semakin tinggi nilai **tool wear** dan **torque**, umumnya meningkatkan
          risiko failure.
        - Perbedaan **`Type`** mesin juga memengaruhi pola risiko.
        """
    )

    # 3. ROC & Confusion Matrix
    st.markdown("---")
    st.markdown("### 3. ROC Curve & Confusion Matrix")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("<h4 style='text-align:center;'>ROC Curve</h4>", unsafe_allow_html=True)
        if IMG_ROC_PATH.exists():
            st.image(
                str(IMG_ROC_PATH),
                caption="ROC Curve - XGBoost (Test Set)",
                use_container_width=True,
            )
            st.caption("Model menunjukkan kemampuan klasifikasi yang sangat baik (AUC ~0.95).")
        else:
            st.info(
                "File ROC Curve belum ditemukan. "
                "Pastikan `images/roc_xgb_test.png` ada di folder proyek."
            )

    with col_cm:
        st.markdown("<h4 style='text-align:center;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        if IMG_CM_PATH.exists():
            st.image(
                str(IMG_CM_PATH),
                caption="Confusion Matrix - XGBoost (Test Set)",
                use_container_width=True,
            )
            st.caption("Model mendeteksi sebagian besar mesin gagal meski masih menghasilkan false positive.")
        else:
            st.info(
                "File Confusion Matrix belum ditemukan. "
                "Pastikan `images/cm_xgb_test.png` ada di folder proyek."
            )

    # 4. Cumulative Gain & Lift
    st.markdown("---")
    st.markdown("### 4. Cumulative Gain & Lift Chart")

    col_gain, col_lift = st.columns(2)

    with col_gain:
        st.markdown("<h4 style='text-align:center;'>Cumulative Gain</h4>", unsafe_allow_html=True)
        if IMG_CUM_GAIN.exists():
            st.image(
                str(IMG_CUM_GAIN),
                caption="Cumulative Gain - XGBoost (Test Set)",
                use_container_width=True,
            )
            st.caption("Model menangkap mayoritas failure hanya dari sedikit mesin berisiko teratas.")
        else:
            st.info(
                "File cumulative gain belum ditemukan. "
                "Pastikan `images/cum_gain_xgb_test.png` ada di folder proyek."
            )

    with col_lift:
        st.markdown("<h4 style='text-align:center;'>Lift Chart</h4>", unsafe_allow_html=True)
        if IMG_LIFT.exists():
            st.image(
                str(IMG_LIFT),
                caption="Lift Chart - XGBoost (Test Set)",
                use_container_width=True,
            )
            st.caption("Model memberikan lift sangat tinggi pada mesin dengan risiko tertinggi.")
        else:
            st.info(
                "File lift chart belum ditemukan. "
                "Pastikan `images/lift_xgb_test.png` ada di folder proyek."
            )

    # 5. SHAP Summary
    st.markdown("---")
    st.markdown("### 5. SHAP Summary Plot")

    if IMG_SHAP_SUMMARY.exists():
        st.image(
            str(IMG_SHAP_SUMMARY),
            caption="SHAP Summary Plot (Top Features)",
            use_container_width=True,
        )
        st.caption("Nilai tinggi pada Torque, Tool Wear, dan Air Temperature paling mendorong failure.")
    else:
        st.info(
            "File SHAP Summary belum ditemukan. "
            "Pastikan `images/shap_summary.png` ada di folder proyek."
        )

    # 6. SHAP Dependence
    st.markdown("### 6. SHAP Dependence Plot – Fitur Utama")

    shap_options = {
        "Torque (Nm)": IMG_SHAP_TORQUE,
        "Tool Wear (min)": IMG_SHAP_TOOL_WEAR,
        "Air Temperature (K)": IMG_SHAP_AIR_TEMP,
    }

    shap_captions = {
        "Torque (Nm)": "Torque tinggi memberi kontribusi terbesar terhadap peningkatan risiko gagal.",
        "Tool Wear (min)": "Tool Wear tinggi secara konsisten meningkatkan risiko failure.",
        "Air Temperature (K)": "Risiko failure naik tajam pada Air Temperature yang tinggi.",
    }

    selected = st.selectbox(
        "Pilih fitur untuk melihat hubungan nilai fitur dengan kontribusi SHAP:",
        list(shap_options.keys()),
    )

    shap_path = shap_options[selected]
    if shap_path.exists():
        st.image(
            str(shap_path),
            caption=f"SHAP Dependence Plot – {selected}",
            use_container_width=True,
        )
        st.caption(shap_captions[selected])
    else:
        st.info(
            f"File untuk {selected} belum ditemukan. "
            "Pastikan nama file di folder `images/shap_dependence/` sudah sesuai."
        )