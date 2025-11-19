# app.py
import streamlit as st
import pandas as pd
import time 

# Impor fungsi kustom kita
import preprocessing as pp
import model_utils as mu

# --- Konfigurasi Halaman & Session State ---
st.set_page_config(
    page_title="DSS Live Training (Single File)",
    layout="wide"
)

# Session state untuk menyimpan data dan status
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'association_rules' not in st.session_state:
    st.session_state.association_rules = None
if 'antecedents' not in st.session_state:
    st.session_state.antecedents = None

# =============================================================================
# --- SIDEBAR (Input Pengguna) ---
# =============================================================================

st.sidebar.title("🚀 Pengaturan DSS (Live Training)")

# --- 1. Upload Dataset (Hanya SATU file) ---
st.sidebar.header("1. Upload Data Gabungan")
uploaded_file = st.sidebar.file_uploader("Upload file CSV (Gabungan)", type="csv")

if uploaded_file and not st.session_state.data_loaded:
    with st.spinner("Memuat dan membersihkan data..."):
        data = pp.load_and_preprocess_data(uploaded_file)
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.sidebar.success("Data berhasil dimuat!")

if not st.session_state.data_loaded:
    st.info("Silakan unggah file CSV gabungan di sidebar untuk memulai.")
    st.stop()

# --- 2. Pemilihan Kolom Kunci & Fitur ---
st.sidebar.header("2. Konfigurasi Kolom")
df = st.session_state.data
all_columns = df.columns.tolist()

# ... (fungsi find_default tetap sama) ...
def find_default(cols, names):
    for name in names:
        if name in cols:
            return cols.index(name)
    return 0 # Default ke kolom pertama jika tidak ditemukan

key_col = st.sidebar.selectbox(
    "Kolom Kunci (ID Customer/Household)", 
    all_columns, 
    index=find_default(all_columns, ['customer_id', 'household_key', 'user_id'])
)
product_list_col = st.sidebar.selectbox(
    "Kolom List Produk (harus format string list)", 
    all_columns, 
    index=find_default(all_columns, ['product_list', 'products', 'items'])
)
# ... (available_features tetap sama) ...
# --- PERBAIKAN PENTING ---
# Pastikan 'basket_id' atau ID numerik lainnya juga dikecualikan
# dari 'available_features' jika ada.
numeric_ids = ['basket_id', 'BASKET_ID', 'transaction_id'] 
exclude_cols = [key_col, product_list_col, 'PX'] + numeric_ids

available_features = [
    col for col in all_columns 
    if col not in exclude_cols
]

st.sidebar.subheader("Pilih Fitur Demografi/Profiling")
demo_features = st.sidebar.multiselect(
    "Fitur yang akan digunakan untuk training:", 
    available_features, 
    default=available_features # Default: pilih semua yang tersedia
)

# --- REVISI: Langkah 2B: Jalankan Association Rules ---
st.sidebar.header("Langkah 1: Jalankan Association Rules")

min_support_val = st.sidebar.slider(
    "Minimum Support FPGrowth", 
    min_value=0.001, 
    max_value=0.1, 
    value=0.01, 
    step=0.001,
    format="%.3f",
    help="Support yang lebih rendah = lebih banyak rules, tapi lebih lama."
)

if st.sidebar.button("Jalankan Association Rules (ARM)"):
    with st.spinner(f"Mengonversi kolom '{product_list_col}'..."):
        data_processed_arm = pp.convert_product_list(df.copy(), product_list_col)
    
    with st.spinner(f"Menjalankan FPGrowth (min_support={min_support_val})..."):
        rules, antecedents = pp.run_association_rules(
            data_processed_arm, 
            product_list_col, 
            min_support=min_support_val
        )
        st.session_state.association_rules = rules
        st.session_state.antecedents = antecedents
        st.success("Association Rules selesai!")

# --- 3. Pemilihan Produk Target ---
st.sidebar.header("Langkah 2: Target Prediksi (ANN)")
st.sidebar.info("Pilih target berdasarkan hasil ARM di layar utama, lalu salin/ketik di bawah.")

# Tampilkan daftar antecedents yang ditemukan sebagai bantuan
if st.session_state.antecedents:
    default_antecedent = st.session_state.antecedents[0] if st.session_state.antecedents else ""
    selected_antecedent = st.sidebar.selectbox(
        "Atau pilih dari antecedent populer:",
        options=[""] + st.session_state.antecedents,
        help="Memilih ini akan mengisi box di bawah."
    )
    
    if selected_antecedent:
        target_products_str_default = selected_antecedent
    else:
        target_products_str_default = ""
else:
    target_products_str_default = ""

target_products_str = st.sidebar.text_input(
    "Masukkan Produk Target (pisahkan koma)", 
    target_products_str_default
)
target_list = set(p.strip().upper() for p in target_products_str.split(',') if p)

# --- TAMBAHAN: Pilihan Metode Resampling ---
resampling_method = st.sidebar.selectbox(
    "Metode Resampling Data Latih:",
    options=['undersampling', 'oversampling'],
    index=1, # <-- Default baru ke 'oversampling' (SMOTE)
    format_func=lambda x: "SMOTE (Oversampling)" if x == 'oversampling' else "Random (Undersampling)",
    help="Oversampling (SMOTE) membuat data minoritas sintetis (disarankan). Undersampling (RUS) membuang data mayoritas."
)


# --- 4. Tombol Proses (ANN) ---
if st.sidebar.button("TRAIN ANN MODEL", type="primary"):
    
    if not target_list:
        st.error("Harap tentukan Produk Target untuk ANN (di Langkah 2).")
        st.stop()
        
    st.session_state.model = None
    
    # 1. Konversi Kolom Product List (SANGAT PENTING)
    with st.spinner(f"Mengonversi kolom '{product_list_col}'..."):
        data_processed_ann = pp.convert_product_list(df.copy(), product_list_col)

    # 2. Buat Variabel Target PX
    with st.spinner(f"Membuat variabel target 'PX' untuk {target_products_str}..."):
        data_with_target = pp.create_target_variable(
            data_processed_ann, product_list_col, target_list
        )
    
    # 3. Encode Fitur
    with st.spinner("Melakukan encoding fitur demografi..."):
        # --- PERBAIKAN LOGIKA X_DATA ---
        
        # Simpan kolom asli SEBELUM encoding
        original_cols = set(data_with_target.columns)
        
        data_encoded = pp.encode_features(data_with_target, demo_features)
            
        y_data = data_encoded['PX']
        
        # Dapatkan daftar kolom fitur yang SUDAH di-encode
        encoded_cols = set(data_encoded.columns)
        
        # Ini adalah fitur demografi kita yang baru di-encode
        # (e.g., 'AGE_DESC_25-34', 'INCOME_DESC_50-75K')
        new_encoded_features = list(encoded_cols - original_cols)

        if not new_encoded_features:
            st.error("Tidak ada fitur demografi yang di-encode. Apakah Anda sudah memilih fitur di sidebar? Model tidak bisa dilatih.")
            st.stop()

        # Buat X_data HANYA dari fitur-fitur yang di-encode
        X_data = data_encoded[new_encoded_features].copy()
        
        st.session_state.X_full = X_data
        st.session_state.y_full = y_data
        
        # Simpan data asli (non-encode) untuk tampilan hasil akhir
        st.session_state.full_data_with_keys = data_encoded[[key_col, product_list_col, 'PX']]

    # 4. Split dan Resample
    with st.spinner("Melakukan Train-Test Split dan Resampling..."):
        X_train, y_train, X_test, y_test = mu.split_and_resample(
            st.session_state.X_full, 
            st.session_state.y_full,
            method=resampling_method # <-- Kirim metode yang dipilih
        )
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        # --- Tambahan Debugging ---
        st.subheader("Info Debugging (Setelah Resampling)")
        st.write("Distribusi Y_Train (Data Latih - Setelah Resampling):")
        st.dataframe(pd.DataFrame(y_train.value_counts()))
        st.write("Distribusi Y_Test (Data Uji - Asli/Imbalanced):")
        st.dataframe(pd.DataFrame(y_test.value_counts()))
        st.write("Info X_Train (Fitur Latih):")
        st.dataframe(X_train.head())
        X_train.info(buf=st)


    # 5. Latih Model ANN (Ini adalah proses yang LAMA)
    with st.spinner(f"MELATIH MODEL ANN... (Bisa memakan waktu)"):
        start_time = time.time()
        model = mu.train_ann_model(X_train, y_train)
        end_time = time.time()
        
        st.session_state.model = model 
        st.sidebar.info(f"Pelatihan ANN selesai dalam {end_time - start_time:.2f} dtk.")

    # 6. Evaluasi Model (pada Test Set Asli)
    with st.spinner("Menghasilkan metrik evaluasi..."):
        st.session_state.eval_metrics = mu.generate_evaluation_metrics(
            model, 
            st.session_state.X_test, 
            st.session_state.y_test
        )

    # 7. Buat Prediksi untuk SEMUA Data (untuk tabel hasil)
    with st.spinner("Membuat prediksi untuk semua data..."):
        full_probs, full_preds = mu.get_predictions(
            model, 
            st.session_state.X_full
        )
        
        results_df = st.session_state.full_data_with_keys.copy()
        results_df['Probabilitas_Beli_PX'] = full_probs
        results_df['Prediksi_Beli_PX'] = full_preds
        st.session_state.prediction_results = results_df

    st.success("Pelatihan ANN dan Prediksi Selesai!")

# =============================================================================
# --- MAIN AREA (Tampilan Hasil) ---
# =============================================================================

st.title("📊 DSS - Association Rules & ANN")

# --- TAMPILAN BARU: Hasil Association Rules ---
if st.session_state.association_rules is not None:
    st.header("Hasil Langkah 1: Association Rules (ARM)")
    if st.session_state.association_rules.empty:
        st.warning("Tidak ada rules yang ditemukan dengan pengaturan support/lift saat ini.")
    else:
        st.info(f"Ditemukan {len(st.session_state.association_rules)} aturan menarik. Gunakan tabel ini untuk menentukan target ANN Anda di sidebar (Langkah 2).")
        
        # Tampilkan kolom yang sudah diformat string
        display_cols = [
            'antecedents_str', 
            'consequents_str', 
            'support', 
            'confidence', 
            'lift'
        ]
        # Pastikan kolom-kolom ini ada sebelum menampilkannya
        valid_cols = [col for col in display_cols if col in st.session_state.association_rules.columns]
        st.dataframe(st.session_state.association_rules[valid_cols])
    
    st.markdown("---") # Pemisah visual
    

# --- TAMPILAN LAMA: Hasil ANN ---
if 'model' not in st.session_state or st.session_state.model is None:
    if st.session_state.association_rules is None:
        st.info("Silakan konfigurasikan input di sidebar kiri dan klik 'Jalankan Association Rules (ARM)' (Langkah 1).")
    else:
        st.info("Sekarang, pilih target Anda dari hasil ARM di atas, masukkan di sidebar, lalu klik 'TRAIN ANN MODEL' (Langkah 2).")

else:
    # --- 5. Tampilan Hasil Evaluasi Model ---
    st.header("Hasil Langkah 2: Evaluasi Model ANN (diuji pada Test Set)")
    eval_data = st.session_state.eval_metrics
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC-ROC Score", f"{eval_data['auc']:.4f}")
        st.subheader("Classification Report (Test Set)")
        st.dataframe(pd.DataFrame(eval_data['report']).transpose())

    with col2:
        st.subheader("Confusion Matrix (Test Set)")
        st.pyplot(eval_data['cm_plot'])
    
    st.subheader("ROC Curve (Test Set)")
    st.pyplot(eval_data['roc_plot'])

    # --- 6. Prediksi Produk untuk Setiap Household (Full Data) ---
    st.header("Prediksi ANN pada Keseluruhan Data")
    st.dataframe(st.session_state.prediction_results)
    
    @st.cache_data
    def convert_df_to_csv(_df):
        return _df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(st.session_state.prediction_results)
    st.download_button(
        label="📥 Download Hasil Prediksi (CSV)",
        data=csv_data,
        file_name="prediksi_live_model.csv",
        mime="text/csv",
    )