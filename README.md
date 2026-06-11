# 💳 Credit Scoring - CI/CD & Monitoring - German Credit Dataset

Repository ini merupakan proyek _submission_ akhir untuk mata kuliah **Membangun Sistem Machine Learning** (Dicoding). Fokus utama proyek ini adalah menerapkan prinsip **MLOps** dalam membangun pipeline _machine learning_ yang _end-to-end_ dan terotomatisasi.

**Fitur Utama & Cakupan Proyek:**

- **Dataset:** Menggunakan _German Credit Dataset_ untuk kasus _Credit Scoring_.
- **Continuous Integration (CI):** Otomatisasi pengujian kode dan integrasi pipeline.
- **Monitoring:** Sistem pemantauan performa model secara berkala saat berjalan.

## 🎯 Tujuan Proyek

Membangun sistem prediksi _credit scoring_ yang handal dengan implementasi:

- **Experiment Tracking:** Manajemen eksperimen menggunakan **MLflow** yang di-host di **DagsHub**.
- **Continuous Integration (CI):** Otomatisasi pengujian kode dan integrasi pipeline menggunakan **GitHub Actions**.
- **Automated Deployment:** Proses _automated model training_ serta _Docker image build & push_ ke **Docker Hub**.

## 🛠️ Fitur Utama

- **Automated Pipeline:** Proses _data preprocessing_ dan _hyperparameter tuning_ (menggunakan `GridSearchCV`) yang berjalan otomatis.
- **Experiment Logging:** Pencatatan manual parameter, metrik, dan _artifact_ ke MLflow.
- **CI Pipeline (GitHub Actions):** Menjalankan _training_ otomatis, melakukan pengujian kode, serta melakukan _build & push_ Docker image secara otomatis ke Docker Hub.
- **Model Inference:** Penyediaan API untuk prediksi real-time menggunakan **FastAPI** dan **MLflow**.
- **Monitoring Stack:**
  - **Prometheus:** Pengumpulan metrik performa sistem dan model (_metrics collection_).
  - **Grafana:** Visualisasi data dan _dashboard_ pemantauan yang interaktif.
- **Alerting System:** Dilengkapi dengan 3 _alert rules_ untuk mendeteksi anomali atau penurunan performa sistem.
- **Containerized Deployment:** Seluruh layanan dibungkus dan dijalankan dengan mudah menggunakan **Docker Compose**.

## 📁 Struktur Proyek

```text
Workflow-CI/
├── .github/workflows/
│   └── ci.yml
├── MLProject/
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── preprocessor.pkl
│   └── German-Credit-Dataset/
│       ├── german_credit_train_preprocessed.csv
│       └── german_credit_test_preprocessed.csv
├── docker-compose.yml
├── Dockerfile
├── inference.py
├── prometheus.yml
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Jalankan Monitoring Stack

```bash
docker compose up --build -d
```

**Akses:**

- **Inference API:** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

### 2. Dashboard Grafana

Dashboard sudah dibuat dengan **10 panel** yang mencakup:

- HTTP Request Rate
- Prediction Performance (Good vs Bad)
- Prediction Latency (p95)
- Error Rate
- Active Requests
- Model Load Status
- Requests by Endpoint

---

## 🔔 Alert Rules

Terdapat 3 alert rules yang telah dikonfigurasi:

| Alert Name              | Kondisi                  | Severity |
| :---------------------- | :----------------------- | :------- |
| Model Down              | `model_load_status == 0` | Critical |
| High 5xx Error Rate     | Error 5xx muncul         | Warning  |
| High Prediction Latency | p95 latency > 1 detik    | Warning  |

---

## 📊 Monitoring

Semua metrik diekspos melalui endpoint `/metrics` pada service inference dan discrape oleh Prometheus.

---

## 👤 Profil Pengembang

Proyek ini dikembangkan oleh **Muhammad Rafi Aditya** sebagai bagian dari kurikulum _Membangun Sistem Machine Learning (Level Mahir)_ di **Dicoding Indonesia** dalam program Pijak in collaboration with IBM SkillsBuild.
