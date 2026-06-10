# 💳 Credit Scoring MLOps - German Credit Dataset

Proyek ini adalah submission akhir untuk mata kuliah **Membangun Sistem Machine Learning** (Dicoding). Fokus utama proyek ini adalah menerapkan prinsip **MLOps** dalam membangun pipeline machine learning yang _end-to-end_ dan terotomatisasi.

## 🎯 Tujuan Proyek

Membangun sistem prediksi _credit scoring_ yang handal menggunakan dataset **German Credit** dengan implementasi:

- Experiment tracking menggunakan **MLflow** yang di-host di **DagsHub**
- Continuous Integration (CI) menggunakan **GitHub Actions**
- Automated model training + Docker image build & push ke **Docker Hub**

## 🛠️ Fitur Utama

- Automated data preprocessing
- Hyperparameter tuning menggunakan `GridSearchCV`
- Manual logging ke MLflow (parameter, metrik, dan artifact)
- Continuous Integration yang menjalankan training otomatis
- Docker image otomatis di-build dan di-push ke Docker Hub via GitHub Actions

## 📁 Struktur Proyek

```text
Workflow-CI/
├── .github/workflows/
│   └── ci.yml
├── MLProject/
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   └── German-Credit-Dataset/
│       ├── german_credit_train_preprocessed.csv
│       └── german_credit_test_preprocessed.csv
├── Dockerfile
├── app.py
├── requirements.txt
└── README.md (opsional)
```

## 🚀 Memulai (Getting Started)

### 1. Eksekusi Lokal

Pastikan Anda memiliki Python 3.x, lalu jalankan perintah berikut:

```bash
cd MLProject
pip install -r requirements.txt
python modelling_tuning.py
```

### 2. Automasi via GitHub Actions

Pipeline ini sudah dikonfigurasi untuk berjalan otomatis. Cukup lakukan:

1. `git add .`
2. `git commit -m "Update model logic"`
3. `git push origin main`
   _Workflow akan mentrigger training otomatis dan mengirimkan hasilnya ke DagsHub, dilanjutkan dengan proses build serta push Docker image ke Docker Hub._

## 📦 Docker Image

Artifact berupa Docker image dari pipeline CI terupdate saat ini sudah tersedia secara publik pada tautan Docker Hub berikut:
👉 **[Docker Image Repository](https://hub.docker.com/repository/docker/rfdtyaaa/credit-scoring-api/general)**

## 📊 Hasil Eksperimen

Hasil tracking, perbandingan metrik, dan model artifact dapat diakses secara publik melalui dashboard berikut:
👉 **[DagsHub MLflow Dashboard](https://dagshub.com/RFDTYAA/credit-scoring-mlops.mlflow)**

---

## 👤 Profil Pengembang

Proyek ini dikembangkan oleh **Muhammad Rafi Aditya** sebagai bagian dari kurikulum _Membangun Sistem Machine Learning (Level Mahir)_ di **Dicoding Indonesia** dalam program Pijak in collaboration with IBM SkillsBuild.
