# Laporan Proyek Machine Learning - Heriswaya

## Domain Proyek

Dalam dunia industri ritel, khususnya dalam kategori produk seperti Elektronik, manajemen rantai pasok dan stok barang menjadi salah satu aspek yang paling krusial. Permintaan yang tidak terprediksi dengan baik dapat menyebabkan dua risiko besar: *stock-out* (kehabisan stok) atau *overstock* (kelebihan stok). Kedua kondisi ini dapat mengakibatkan kerugian finansial baik dalam bentuk kehilangan peluang penjualan maupun biaya penyimpanan tambahan.<br>
Untuk mengatasi hal ini, perusahaan membutuhkan sistem yang mampu **memperkirakan permintaan barang secara akurat** dalam jangka waktu tertentu. Salah satu pendekatan yang relevan dan powerful untuk menangani masalah ini adalah dengan memanfaatkan teknik **Machine Learning (ML)**, terutama pendekatan regresi berbasis time series.<br>
Forecasting permintaan (demand forecasting) merupakan bagian dari sistem **demand planning**, dan telah terbukti memiliki dampak signifikan dalam meningkatkan efisiensi logistik dan keuntungan [1]. Berbagai studi seperti oleh Fildes et al. (2008) dan Hyndman & Athanasopoulos (2018) menunjukkan bahwa metode ML berbasis feature engineering memiliki keunggulan dalam memodelkan pola permintaan yang kompleks dan non-linear.<br>
Dengan adanya digitalisasi data historis penjualan dan informasi tanggal, perusahaan kini dapat melatih model prediktif untuk memperkirakan permintaan ke depan. Proyek ini hadir untuk mendemonstrasikan kemampuan model prediktif berbasis ML untuk memperkirakan permintaan kategori produk elektronik.<br>
**Referensi**:<br>
- [1] Fildes, R., Goodwin, P., Lawrence, M., & Nikolopoulos, K. (2008). Effective Forecasting and Judgmental Adjustments: An Empirical Evaluation and Strategies for Improvement. International Journal of Forecasting, 24(1), 3–26.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice (2nd ed.). OTexts. https://otexts.com/fpp2/

## Business Understanding

Dalam konteks operasional perusahaan, kemampuan memprediksi permintaan produk sangat penting untuk efisiensi logistik, pengadaan, dan penjualan. Proyek ini dibangun berdasarkan kebutuhan nyata perusahaan untuk menyeimbangkan persediaan dan permintaan secara tepat waktu.

### Problem Statements

- **Pernyataan Masalah 1** : Bagaimana memprediksi permintaan produk elektronik secara akurat berdasarkan data historis penjualan?
- **Pernyataan Masalah 2** : Bagaimana mengembangkan model Machine Learning yang dapat menangkap pola waktu seperti tren musiman, siklus mingguan, dan dependensi waktu (lag)?
- **Pernyataan Masalah 3** : Bagaimana mengevaluasi performa model prediksi permintaan terhadap data pelatihan dan data pengujian?

### Goals

- **Tujuan 1** : Menghasilkan prediksi permintaan produk elektronik menggunakan model yang telah dilatih.
- **Tujuan 2** : Membuat fitur-fitur time series (seperti lag dan informasi kalender) yang informatif dan relevan untuk model ML.
- **Tujuan 3** : Mengukur performa model dengan metrik evaluasi yang sesuai (seperti MAE, RMSE, R²) baik pada data pelatihan maupun pengujian.

### Solution statements
- **Solusi 1**: Mengembangkan pipeline machine learning berbasis regresi menggunakan model seperti Random Forest, XGBoost, dan sejenisnya.
- **Solusi 2**: Menerapkan feature engineering berbasis waktu (lag features dan kalender) agar model dapat mengenali pola time series yang tidak eksplisit.
- **Solusi 3**: Melakukan evaluasi model menggunakan metrik kuantitatif: MAE (Mean Absolute Error), RMSE (Root Mean Square Error), dan R² (coefficient of determination) pada data training dan testing.
- **Solusi 4**: Mengimplementasikan hyperparameter tuning supaya modelnya lebih maksimal.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data toko `S001` dan semua jenis kateogri. Lebih lengkapnya:
> This dataset follows the format of the Retail Store Inventory Forecasting Dataset and corrects mislabeled entries such as store and product IDs. Additionally, it includes an Epidemic feature to simulate retail conditions during the COVID-19 pandemic period, enhancing the realism and practical value of the data. These improvements are aimed at making the dataset more suitable for time series forecasting tasks.

Dataset berasal dari `Kaggle`, yang bisa diakses pada link berikut. [Retail Inventory Demand Forecasting](https://www.kaggle.com/code/devraai/retail-inventory-demand-forecasting).

Berikut adalah deskripsi variabel dalam dataset:

### Variabel-variabel pada Dataset Retail Inventory Demand Forecasting:

| Nama Kolom (English)   | Nama Kolom (Indonesia) | Deskripsi Lengkap |
|------------------------|------------------------|------------------|
| **Date**               | Tanggal                | Tanggal pencatatan data transaksi/harian. Kolom ini merupakan variabel temporal yang kritikal untuk analisis time series. |
| **Store ID**           | ID Toko                | Pengidentifikasi unik untuk setiap toko. Digunakan untuk melacak performa masing-masing outlet. |
| **Product ID**         | ID Produk              | Kode unik produk. Memungkinkan pelacakan item spesifik di seluruh toko. |
| **Category**           | Kategori Produk        | Klasifikasi produk (mis: Elektronik, Pakaian, Makanan). Penting untuk analisis segmentasi. |
| **Region**             | Wilayah                | Lokasi geografis toko (mis: Jawa Barat, Sumatera Utara). Untuk analisis regional. |
| **Inventory Level**    | Tingkat Persediaan     | Jumlah unit yang tersedia di stok pada hari tersebut. Indikator manajemen inventaris. |
| **Units Sold**         | Unit Terjual           | Jumlah produk yang terjual pada hari rekaman. Variabel target utama untuk analisis penjualan. |
| **Units Ordered**      | Unit Dipesan           | Jumlah produk yang dipesan untuk restok. Menunjukkan aktivitas rantai pasokan. |
| **Price**              | Harga                  | Harga jual produk per unit (dalam mata uang lokal). |
| **Discount**           | Diskon                 | Besaran diskon yang diterapkan (jika ada). Dinyatakan dalam persentase atau nilai absolut. |
| **Weather Condition**  | Kondisi Cuaca          | Deskripsi kondisi cuaca (mis: Hujan, Cerah). Memengaruhi pola pembelian. |
| **Promotion**          | Promosi                | Indikator biner (1/0) menunjukkan apakah produk sedang dalam promosi. |
| **Competitor Pricing** | Harga Kompetitor       | Harga produk sejenis dari pesaing. Untuk analisis harga kompetitif. |
| **Seasonality**        | Musim                  | Kategori musim (Mis: Musim Hujan, Kemarau). Memodelkan pola musiman. |
| **Epidemic**           | Wabah                  | Indikator biner (1/0) menandakan apakah terjadi wabah penyakit yang memengaruhi penjualan. |
| **Demand**             | Permintaan             | Perkiraan permintaan harian produk. Sering digunakan sebagai variabel target dalam forecasting. |

### Catatan Analitis:
1. **Variabel Temporal**: 
   - `Tanggal` menjadi dasar untuk analisis deret waktu (time series analysis).
   
2. **Variabel Kategorikal**: 
   - `Kategori Produk`, `Wilayah`, `Kondisi Cuaca`, dan `Musim` memerlukan encoding sebelum pemodelan.

3. **Indikator Eksternal**: 
   - `Harga Kompetitor` dan `Wabah` merupakan faktor eksternal yang memengaruhi permintaan.

4. **Metrik Kinerja**: 
   - Rasio `Unit Terjual` terhadap `Tingkat Persediaan` dapat mengindikasikan efisiensi stok.

### Exploratory Data Analysis (EDA)

Beberapa langkah eksplorasi data dilakukan untuk memahami karakteristik dataset:

* **Plot time series** untuk melihat pola tren jangka panjang dan fluktuasi musiman.
* **Distribusi nilai demand** menunjukkan bahwa sebagian besar nilai berada pada kisaran tertentu dengan sedikit outlier.
* **Correlation Plot (Heatmap)** untuk melihat korelasi antar variabel dan sangat membantu dalam fitur yang akan digunakan.

## Data Preparation

Tahapan data preparation dilakukan secara berurutan untuk memastikan data siap digunakan dalam pelatihan model Machine Learning. Berikut adalah langkah-langkah yang dilakukan:

1. **Konversi Tanggal**
   Kolom `date` dikonversi ke format datetime agar dapat digunakan untuk ekstraksi fitur waktu dan manipulasi indeks.

2. **Sortir Data**
   Data diurutkan berdasarkan tanggal agar konsisten secara kronologis sebelum dilakukan proses pembuatan fitur lag.

3. **Pembuatan Fitur Lag**
   Dibuat fitur-fitur lag seperti `lag_1`, `lag_2`, `lag_3`, dst. Tujuannya adalah agar model dapat belajar dari permintaan sebelumnya, sesuai dengan sifat data time series.

4. **Ekstraksi Fitur Kalender**
   Ditambahkan fitur-fitur berdasarkan waktu, seperti:

   * `Year` (tahun),
   * `Month` (bulan),
   * `Day` (tanggal dalam bulan).

   Fitur-fitur ini membantu model memahami pola musiman dan siklus mingguan.

5. **Handling Missing Values**
   Setelah penambahan fitur lag, akan muncul missing values pada awal data. Baris-baris ini dihapus karena tidak bisa digunakan dalam pelatihan model.

6. **Splitting Dataset**
   Dataset dibagi menjadi dua bagian:

   * **Training Set (80%)**: Data hingga tanggal tertentu digunakan untuk melatih model.
   * **Testing Set (20%)**: Data setelah tanggal tersebut digunakan untuk mengevaluasi model.

Alasan dari tiap tahapan ini adalah agar data dapat direpresentasikan secara optimal ke dalam bentuk fitur yang dapat dikenali oleh model prediktif, khususnya dalam memodelkan pola time series non-linier dan musiman.

## Modeling

Proyek ini menggunakan algoritma **XGBoost Regressor** untuk mem-forecast permintaan harian masing-masing kategori produk di toko `S001`.
XGBoost dipilih karena:

| Kelebihan                                                                                                                                                 | Kekurangan                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| • Mampu menangkap hubungan non-linear <br>• Mendukung feature importance & interpretabilitas <br>• Scalable dan cepat berkat optimisasi gradient-boosting | • Lebih banyak hyper-parameter sehingga rentan overfitting <br>• Memerlukan tuning yang hati-hati agar generalisasi baik |

### 1. Pipeline Pemodelan

| Langkah                   | Penjelasan ringkas                                                                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Feature Engineering**   | `lag_1 – lag_3`, serta fitur kalender (`Day`, `Month`, `Year`, `DayOfWeek`).                                                                                                                                             |
| **Standarisasi**          | `StandardScaler` dipelajari di training set; scaler disimpan per-kategori.                                                                                                                                               |
| **Train-Test Split**      | 80 % data pertama → *train* (chronological), 20 % terakhir → *test*.                                                                                                                                                     |
| **Baseline Model**        | XGBoost default `n_estimators=100, learning_rate=0.1, max_depth=5, subsample=1.0, colsample_bytree=1.0`.                                                                                                                 |
| **Hyperparameter Tuning** | `GridSearchCV` 3-fold pada *train* set dengan grid:<br>`max_depth∈{3,5,7}`, `learning_rate∈{0.01,0.1,0.2}`, `n_estimators∈{100,200}`, `subsample∈{0.8,1.0}`, `colsample_bytree∈{0.8,1.0}`. <br>Skor optimasi : **−MAE**. |
| **Model Terbaik**         | Model dengan kombinasi parameter paling kecil MAE dicatat & dilatih ulang pada seluruh *train* set, lalu dievaluasi.                                                                                                     |

**Parameter terbaik per-kategori** (ringkasan):

| Kategori    | `max_depth` | `n_estimators` | `learning_rate` | `subsample` | `colsample_bytree` |
| ----------- | :---------: | :------------: | :-------------: | :---------: | :----------------: |
| Electronics |      7      |       200      |       0.1       |     0.8     |         0.8        |
| Clothing    |      7      |       200      |       0.1       |     1.0     |         0.8        |
| Groceries   |      7      |       200      |       0.1       |     1.0     |         0.8        |
| Toys        |      5      |       100      |       0.1       |     1.0     |         0.8        |
| Furniture   |      5      |       200      |       0.1       |     0.8     |         1.0        |

> **Why XGBoost (tuned) chosen as final model?**
> Dibanding baseline, tuning menurunkan MAE & RMSE pada 4 dari 5 kategori dan meningkatkan R² test; model lain (RF, Linear) telah dicoba sebelumnya namun memiliki error lebih tinggi. Karena itu XGBoost-tuned dianggap paling layak untuk deployment.

## Evaluation

### 1. Metrik yang Digunakan

| Metrik   |       Rumus                                                                      | Interpretasi                                         |   |                                                      |
| -------- | -------------------------------------------------------------------------- | ---------------------------------------------------- | - | ---------------------------------------------------- |
| **MAE**  | $\displaystyle \frac1n\sum_{i=1}^{n}(y_i-\hat y_i)$ | Rata-rata selisih absolut; makin kecil → makin baik. |
| **RMSE** | $\displaystyle \text{RMSE}= \sqrt{\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2 }$ | Menekankan penalti error besar; unit sama dg target. |   |                                                      |
| **R²**   | $1-\dfrac{\sum (y-\hat y)^2}{\sum (y-\bar y)^2}$                           | Proporsi variasi yang dijelaskan model (0 – 1).      |   |                                                      |

### 2. Hasil Akhir

| Kategori    | MAE ↓     | RMSE ↓    | R² Train ↑ | R² Test ↑ | Δ R² (Train-Test) |
| ----------- | --------- | --------- | ---------- | --------- | ----------------- |
| Electronics | **8.78**  | **11.59** | 0.999      | 0.899     | 0.100             |
| Clothing    | **10.47** | **14.42** | 0.999      | 0.900     | 0.099             |
| Groceries   | **12.23** | **15.55** | 0.991      | 0.872     | 0.119             |
| Toys        | **8.84**  | **11.68** | 0.974      | 0.920     | 0.054             |
| Furniture   | **7.60**  | **9.98**  | 0.981      | 0.900     | 0.081             |

> **Interpretasi:**
>
> * MAE & RMSE turun dibanding baseline, terutama Electronics & Groceries.
> * R² test mendekati 0.90 pada 4 kategori, menandakan model menjelaskan ≥ 90 % variasi permintaan harian.
> * Δ R² masih ≤ 0.12 → overfitting moderat, namun dapat diterima untuk use-case operasi stok.

### 3. Insight & Next Step

* Untuk mereduksi selisih Train/Test lebih lanjut, bisa dicoba:

  * regularisasi tambahan (`gamma`, `min_child_weight`),
  * menambah fitur eksternal (promosi mendatang, cuaca prakiraan),
  * cross-validation berbasis *time-series split*.

Model akhir siap dipakai sebagai komponen **decision-support** bagi tim inventory untuk menentukan jumlah restock per-kategori selama satu bulan mendatang.
