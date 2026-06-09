# 💳 Credit Scoring MLOps - German Credit Dataset

Proyek ini adalah submission akhir untuk mata kuliah **Membangun Sistem Machine Learning** (Dicoding). Fokus utama proyek ini adalah menerapkan prinsip **MLOps** dalam membangun pipeline machine learning yang _end-to-end_ dan terotomatisasi.

## 🎯 Tujuan Proyek

Membangun sistem prediksi _credit scoring_ yang handal menggunakan dataset **German Credit** dengan implementasi:

- **Experiment Tracking**: Pencatatan eksperimen yang rapi menggunakan **MLflow** yang di-hosting di **DagsHub**.
- **Continuous Integration (CI)**: Automasi training dan validasi model menggunakan **GitHub Actions**.
- **Automated Training**: Model akan dilatih ulang secara otomatis setiap kali ada perubahan pada kode sumber.

## 🛠️ Fitur Utama

- **Automated Preprocessing**: Pembersihan data otomatis (disimpan dalam format `.pkl`).
- **Hyperparameter Tuning**: Optimasi model **RandomForestClassifier** menggunakan **GridSearchCV**.
- **Comprehensive Logging**:
  - Parameter terbaik & metrik evaluasi (Accuracy, F1-Score, ROC-AUC).
  - Visualisasi otomatis (Confusion Matrix & Feature Importance).
  - Penyimpanan Model Artifact secara terpusat.

## 📁 Struktur Proyek

```text
.
├── .github/workflows/
│   └── ci.yml             # Konfigurasi GitHub Actions
├── MLProject/
│   ├── German-Credit-Dataset/  # Dataset & Preprocessor artifact
│   ├── modelling_tuning.py     # Script utama training & tuning
│   ├── requirements.txt        # Dependensi Python
│   ├── conda.yaml             # Environment untuk MLflow
│   └── MLProject               # Konfigurasi MLflow Project
└── README.md
```

## 🚀 Memulai (Getting Started)

### 1. Eksekusi Lokal

Pastikan Anda memiliki Python 3.x, lalu jalankan perintah berikut:

```bash
# Masuk ke direktori project
cd MLProject

# Install dependensi
pip install -r requirements.txt

# Jalankan training
python modelling_tuning.py
```

### 2. Automasi via GitHub Actions

Pipeline ini sudah dikonfigurasi untuk berjalan otomatis. Cukup lakukan:

1. `git add .`
2. `git commit -m "Update model logic"`
3. `git push origin main`
   _Workflow akan mentrigger training otomatis dan hasilnya langsung terkirim ke DagsHub._

## 📊 Hasil Eksperimen

Hasil tracking, perbandingan metrik, dan model artifact dapat diakses secara publik melalui dashboard berikut:
👉 **[DagsHub MLflow Dashboard](https://dagshub.com/RFDTYAA/credit-scoring-mlops.mlflow)**

## ⚠️ Catatan Teknis

- **Docker Issue**: Saat ini `mlflow.models.build_docker` memiliki dependensi bawaan ke Python 3.9 yang menyebabkan kendala pada beberapa environment. Untuk sementara, deployment via Docker dinonaktifkan dan fokus dialihkan pada stabilitas **CI Pipeline** dan **Experiment Tracking**.

---

## 👤 Profil Pengembang

Proyek ini dikembangkan oleh **Muhammad Rafi Aditya** sebagai bagian dari kurikulum _Membangun Sistem Machine Learning (Level Mahir)_ di **Dicoding Indonesia**.
