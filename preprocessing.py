# preprocessing.py
import pandas as pd
import ast # Library untuk konversi string-list menjadi list
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Memuat dan melakukan pembersihan dasar pada data yang digabungkan.
    """
    df = pd.read_csv(uploaded_file)
    
    # Salin untuk menghindari SettingWithCopyWarning
    df_cleaned = df.copy()
    
    # Tambahkan logika pembersihan Anda di sini, jika perlu
    # Contoh: Menghapus nilai 'Unknown'
    # df_cleaned = df_cleaned[df_cleaned['HOMEOWNER_DESC'] != 'Unknown']
    
    # Hapus baris di mana data penting (seperti demografi) hilang
    # Ini opsional tapi direkomendasikan
    df_cleaned.dropna(inplace=True) 
    
    return df_cleaned

def convert_product_list(df, product_list_col):
    """
    Mengonversi kolom product_list (yang mungkin berupa string)
    menjadi list Python asli.
    
    Format string yang diharapkan: "['PRODUK A', 'PRODUK B']"
    """
    try:
        # ast.literal_eval aman mengevaluasi string
        # dan mengubahnya menjadi objek Python (list, dict, dll)
        df[product_list_col] = df[product_list_col].apply(ast.literal_eval)
        
        # Verifikasi bahwa hasilnya adalah list
        if not isinstance(df[product_list_col].iloc[0], list):
            raise ValueError("Kolom bukan list setelah konversi.")
            
    except (ValueError, SyntaxError, TypeError) as e:
        st.error(f"""
        Gagal mengonversi kolom '{product_list_col}' menjadi list.
        Pastikan format di CSV Anda adalah string list Python, 
        contoh: "['Roti', 'Susu']"
        
        Error: {e}
        """)
        # Coba cara alternatif (jika formatnya 'Roti,Susu')
        # df[product_list_col] = df[product_list_col].apply(lambda x: [item.strip() for item in str(x).split(',')])
        st.stop()
    
    return df

@st.cache_data
def run_association_rules(_df, product_list_col, min_support=0.01):
    """
    Menjalankan FPGrowth dan Association Rules pada kolom product_list.
    """
    # 1. Dapatkan list of lists dari DataFrame
    # Kita perlu memastikan list tidak kosong
    
    # --- PERBAIKAN ---
    # Baris asli: product_lists = _df[product_list_col][lambda x: x and len(x) > 0].tolist()
    # Baris tersebut menyebabkan ValueError karena ambiguitas boolean.
    # Kita ganti dengan cara .apply() yang lebih eksplisit untuk membuat mask:
    
    # Pastikan itu adalah list DAN memiliki panjang > 0
    mask = _df[product_list_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
    product_lists = _df[product_list_col][mask].tolist()
    
    if not product_lists:
        st.warning("Tidak ada data keranjang (product lists) yang valid untuk diproses.")
        return pd.DataFrame(), []

    # 2. One-Hot Encode (Transformasi ke format FPGrowth)
    te = TransactionEncoder()
    te_ary = te.fit(product_lists).transform(product_lists)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    if df_encoded.empty:
        st.warning("Gagal membuat data encoded untuk FPGrowth.")
        return pd.DataFrame(), []

    # 3. Jalankan FPGrowth
    st.write(f"Menjalankan FPGrowth (Support={min_support}) pada {len(df_encoded)} transaksi...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        st.warning(f"Tidak ada itemset yang memenuhi min_support={min_support}. Coba turunkan nilainya.")
        return pd.DataFrame(), []

    # 4. Buat Rules
    st.write("Menghasilkan association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if rules.empty:
        st.warning("Tidak ada rules yang memenuhi min_lift=1.0.")
        return pd.DataFrame(), []
    
    # 5. Filter & Siapkan "Interesting Rules"
    interesting_rules = rules[
        (rules['confidence'] > 0.5) & 
        (rules['lift'] > 1.2)
    ].sort_values(by='lift', ascending=False)
    
    if interesting_rules.empty:
        st.warning("Tidak ada 'interesting rules' (confidence > 0.5 & lift > 1.2).")
        return rules, [] # Kembalikan semua rules

    # Konversi frozenset ke string yang bisa dibaca untuk UI
    interesting_rules['antecedents_str'] = interesting_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    interesting_rules['consequents_str'] = interesting_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Buat daftar unik antecedents untuk dropdown
    unique_antecedents = interesting_rules['antecedents_str'].unique().tolist()
    
    return interesting_rules, unique_antecedents

def create_target_variable(df, product_list_col, target_product_list):
    """
    Membuat variabel target 'PX' (1 atau 0) berdasarkan
    apakah 'target_product_list' ada di dalam 'product_list_col'.
    """
    # Menggunakan set() untuk pencarian yang jauh lebih cepat
    target_set = set(target_product_list)
    
    def contains_itemset(product_list):
        # Memastikan product_list juga set untuk perbandingan
        if not isinstance(product_list, set):
            product_list = set(product_list)
        # Cek apakah target_set adalah subset dari product_list
        return int(target_set.issubset(product_list))
    
    df['PX'] = df[product_list_col].apply(contains_itemset)
    return df

def encode_features(df, demographic_features):
    """
    Melakukan One-Hot Encoding pada fitur demografi yang dipilih.
    """
    if not demographic_features:
        # Jika tidak ada fitur demografi, kembalikan dataframe apa adanya
        return df
        
    # Gunakan pd.get_dummies untuk encoding
    # drop_first=True untuk menghindari multikolinearitas
    df_encoded = pd.get_dummies(df, columns=demographic_features, drop_first=True)
    
    return df_encoded