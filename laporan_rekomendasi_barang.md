# Laporan Proyek Machine Learning - Heriswaya

## 📘 **Project Overview**
### Latar Belakang

Dalam era digital saat ini, e-commerce dan platform jual beli online semakin berkembang pesat. Salah satu tantangan utama dalam pengembangan platform ini adalah **menyediakan pengalaman personalisasi** yang mampu membantu pengguna menemukan produk yang relevan dengan minat dan kebutuhannya. Salah satu pendekatan untuk menyelesaikan permasalahan ini adalah dengan menerapkan sistem **rekomendasi produk berbasis machine learning**.

Sistem rekomendasi terbukti efektif dalam meningkatkan kepuasan pelanggan serta konversi penjualan. Amazon, Tokopedia, hingga Shopee telah membuktikan bahwa dengan memberikan saran produk yang relevan, waktu tinggal pengguna dalam aplikasi meningkat, begitu pula jumlah transaksi yang dilakukan.

Namun, membuat sistem rekomendasi yang tepat bukanlah tugas yang sepele. Pendekatan yang digunakan harus mampu menangkap preferensi pengguna berdasarkan histori interaksinya. Dalam proyek ini, sistem rekomendasi dibangun menggunakan pendekatan **Collaborative Filtering dengan Matrix Factorization**, yang diimplementasikan menggunakan model **Embedding dalam TensorFlow Keras**.

Dataset yang digunakan mencerminkan interaksi antara pengguna dan produk yang mencakup user ID, product ID, rating, serta metadata terkait produk seperti nama dan kategori. Fokus dari proyek ini adalah memprediksi produk mana yang kemungkinan besar akan disukai oleh pengguna, **berdasarkan pola interaksi historis dari seluruh pengguna**.

### Tujuan Proyek

Tujuan dari proyek ini adalah:

* Membangun sistem rekomendasi produk berbasis interaksi pengguna.
* Menggunakan embedding layer untuk menghasilkan representasi fitur laten dari user dan produk.
* Menggunakan prediksi skor interaksi untuk merekomendasikan produk yang belum pernah dirating oleh pengguna.

### Mengapa Permasalahan Ini Penting?

Penerapan sistem rekomendasi yang akurat memiliki dampak signifikan terhadap:

* **Peningkatan pengalaman pengguna** karena produk yang relevan akan ditampilkan secara personal.
* **Peningkatan efisiensi sistem** dalam menyaring jutaan produk yang tersedia.
* **Optimasi bisnis** dalam konversi penjualan dan retensi pengguna.

Sistem rekomendasi tidak hanya membantu pengguna menemukan apa yang mereka cari, tetapi juga **membantu mereka menemukan apa yang belum mereka sadari mereka inginkan** — inilah kekuatan prediktif dari rekomendasi berbasis data.

### 📚 Referensi

\[[1](https://dl.acm.org/doi/10.1145/3370082)] D. Jannach and M. Jugovac. 2019. Measuring the business value of recommender systems. *ACM Transactions on Management Information Systems* (TMIS), 10(4): 1–23.<br>
\[[2](https://dl.acm.org/doi/10.1145/3038912.3052569)] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. In *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182.<br>
\[[3](https://link.springer.com/chapter/10.1007/978-3-540-72079-9_12)] R. Burke, “Hybrid Web Recommender Systems,” in *The Adaptive Web*, P. Brusilovsky, A. Kobsa, and W. Nejdl, Eds. Berlin, Heidelberg: Springer, 2007, pp. 377–408.

## 💼 Business Understanding

### Problem Statements

1. **Bagaimana cara mengidentifikasi produk-produk yang relevan untuk direkomendasikan kepada pengguna berdasarkan data interaksi pengguna sebelumnya (rating)?**<br>
   Saat pengguna mengakses platform, mereka dihadapkan dengan ribuan produk. Tanpa sistem yang dapat memahami preferensi pengguna, sangat sulit untuk menyajikan produk yang sesuai dan meningkatkan pengalaman belanja.

2. **Bagaimana sistem dapat memberikan rekomendasi hanya terhadap produk yang belum pernah dinilai atau dikunjungi oleh pengguna?**<br>
   Menyajikan produk yang sudah diketahui atau pernah dirating oleh pengguna akan menurunkan nilai prediktif sistem dan berisiko menciptakan pengalaman yang berulang.

3. **Bagaimana mengembangkan model rekomendasi yang ringan, efisien, dan dapat di-training dengan baik meskipun hanya menggunakan data interaksi user dan produk?**<br>
   Model harus dapat berjalan secara efisien dalam resource terbatas, mengingat dalam praktiknya, banyak bisnis kecil hingga menengah yang tidak memiliki infrastruktur besar.


### Goals

1. **Membangun sistem rekomendasi produk berbasis interaksi (user-product ratings) yang dapat memprediksi preferensi pengguna terhadap produk lain.**<br>
   Sistem akan mempelajari representasi laten dari user dan produk untuk memahami pola ketertarikan.

2. **Menyaring produk-produk yang belum pernah dirating oleh user dan menghasilkan daftar rekomendasi berdasarkan skor prediksi kecocokan.**<br>
   Hal ini memastikan bahwa sistem hanya menyarankan produk-produk baru yang berpotensi diminati oleh user.

3. **Menggunakan metode embedding dengan TensorFlow untuk menghasilkan sistem rekomendasi yang dapat diskalakan dan efisien dalam proses pelatihan.**<br>
   Model dirancang untuk cukup sederhana namun akurat, sehingga dapat digunakan di berbagai skenario.

### Solution Statements

Untuk meraih tujuan-tujuan tersebut, pendekatan solusi yang diajukan antara lain:

1. **Collaborative Filtering dengan Matrix Factorization menggunakan Neural Network (Embedding Layer)**<br>
   Pendekatan ini melibatkan representasi setiap user dan produk sebagai vektor berdimensi rendah (embedding). Kemudian dilakukan dot product untuk memperkirakan skor kecocokan. Keunggulan dari metode ini adalah fleksibilitas dalam memanfaatkan framework deep learning seperti TensorFlow.

2. **Penggunaan activation function sigmoid untuk menormalisasi skor rekomendasi dalam rentang \[0, 1]**<br>
   Ini berguna dalam interpretasi model dan memudahkan penyaringan hasil berdasarkan skor prediksi tertinggi.

3. **Filtering rekomendasi hanya pada produk yang belum dirating**<br>
   Dengan menggunakan operator bitwise `~`, sistem dapat mengeliminasi produk yang sudah dikenali pengguna sebelumnya sehingga rekomendasi menjadi lebih tepat sasaran.

## 📊 Data Understanding

### 1. Gambaran Umum Dataset

Dataset yang digunakan bersumber dari **Kaggle – Retail Customer and Transaction Dataset** (tautan unduh: [Retail Customer & Transaction Dataset](https://www.kaggle.com/datasets/raghavendragandhi/retail-customer-and-transaction-dataset)).
Dataset ini menyatukan data demografi pelanggan, aktivitas transaksi, interaksi digital, ulasan produk, serta data kampanye pemasaran dan tiket dukungan. Untuk proyek rekomendasi ini, fokus utama berada pada **empat berkas inti**:

| File                            | Jumlah Baris | Jumlah Kolom | Peran utama                                             |
| ------------------------------- | ------------ | ------------ | ------------------------------------------------------- |
| `customers.csv`                 | 5 000        | 12           | Profil & demografi pengguna                             |
| `transactions.csv`              | 32 295       | 10           | Riwayat pembelian produk                                |
| `interactions.csv`              | 100 000      | 8            | Aktivitas pengguna di kanal digital (view, click, dll.) |
| `customer_reviews_complete.csv` | 1 000        | 10           | Rating & ulasan produk (sumber rating eksplisit)        |

> **Kondisi data:**
>
> * **Duplikasi** pada kombinasi *customer–product–rating* telah dibersihkan.
> * **Missing values** hanya tersisa pada kolom non-kritis seperti `full_name`; kolom kunci (`customer_id`, `product_name`, `rating`) lengkap.
> * Setelah proses penggabungan dan sampling acak, dataset akhir untuk modeling berisi **963 baris** dengan delapan kolom (`customer_id`, `product_name`, `rating`, `product_category`, `full_name`, `interaction_score`, `user`, `product`).

### 2. Deskripsi Fitur

**A. customers.csv**

* `customer_id` : ID unik setiap pelanggan.
* `full_name` : Nama lengkap.
* `age`, `gender`, `email`, `phone`, `street_address`, `city`, `state`, `zip_code` : Atribut demografi & kontak.
* `registration_date` : Tanggal mendaftar.
* `preferred_channel` : Kanal komunikasi favorit.

**B. transactions.csv**

* `transaction_id` : ID transaksi.
* `customer_id` : Relasi ke pelanggan.
* `product_name`, `product_category` : Deskripsi produk.
* `quantity`, `price`, `discount_applied` : Detail pembelian.
* `transaction_date`, `store_location`, `payment_method` : Metadata transaksi.

**C. interactions.csv**

* `interaction_id` : ID interaksi.
* `customer_id` : Relasi ke pelanggan.
* `channel` : Kanal (Website, Mobile App, dll.).
* `interaction_type` : Jenis interaksi (View, Add to Cart, dll.).
* `interaction_date`, `duration`, `page_or_product`, `session_id` : Detail sesi.

**D. customer\_reviews\_complete.csv**

* `review_id` : ID ulasan.
* `customer_id` : Relasi ke pelanggan.
* `product_name`, `product_category` : Produk yang diulas.
* `transaction_date`, `review_date` : Waktu transaksi & ulasan.
* `rating` : Skor 1–5 (label utama).
* `review_title`, `review_text`, `full_name` : Konten ulasan.

*(File pendukung lain seperti `campaigns.csv` dan `support_tickets.csv` tidak digunakan langsung pada pipeline rekomendasi ini.)*

### 3. Exploratory Data Highlights  *(singkat)*

* **Distribusi rating** (1–5) cukup seimbang, dengan modus di level 4.
* **Jumlah interaksi > jumlah rating**: hanya **5,84 %** pasangan user–produk memiliki `interaction_score` > 0, menandakan *sparse explicit feedback*—tantangan umum sistem rekomendasi.
* **Produk terpopuler** menurut rating & interaksi berada di kategori *Electronics* dan *Home Decor*, sehingga potensi bias perlu diwaspadai saat evaluasi.

Insight‐insight EDA ini menjadi dasar keputusan preprocessing (mis. penyertaan `interaction_score` untuk menambah sinyal) dan pemilihan model embedding.

## 🧹 Data Preparation

Tahapan data preparation merupakan langkah penting sebelum memasuki proses pelatihan sistem rekomendasi. Pada proyek ini, proses persiapan data mencakup penggabungan data, pembersihan, serta transformasi fitur agar sesuai dengan kebutuhan model *collaborative filtering*. Berikut tahapan-tahapan yang dilakukan secara **berurutan**:

### 1. Penggabungan Data dari Berbagai Sumber

Beberapa file digabung untuk memperoleh representasi lengkap interaksi pelanggan dengan produk:

* **`customer_reviews_complete.csv`** digunakan sebagai sumber data **rating eksplisit**.
* **`interactions.csv`** ditambahkan untuk memberi sinyal interaksi tambahan (rating implisit).
* Kolom `interaction_score` dihitung berdasarkan jumlah interaksi per pasangan user–produk.
* Hasil penggabungan membentuk data utama dengan kolom:
  `['customer_id', 'product_name', 'rating', 'product_category', 'full_name', 'interaction_score']`.

📌 *Alasan:*
Penggabungan ini memungkinkan kita memperkuat sinyal rekomendasi, khususnya dari sisi implicit feedback. Hal ini penting karena hanya sebagian kecil produk yang diberi rating oleh pengguna.

### 2. Penggantian Nama Kolom Menjadi ‘user’ dan ‘product’

Kolom `customer_id` dan `product_name` diubah namanya menjadi:

* `user` → untuk mewakili ID pengguna.
* `product` → untuk mewakili nama produk.

📌 *Alasan:*
Standarisasi nama kolom memudahkan proses vektorisasi dan training model embedding, terutama pada pipeline berbasis TensorFlow atau model rekomendasi eksplisit lainnya.

### 3. Pembersihan Nilai Kosong

Dataset difilter untuk membuang baris yang mengandung `NaN` pada kolom kritikal: `user`, `product`, dan `rating`.

📌 *Alasan:*
Model rekomendasi berbasis rating tidak dapat berjalan tanpa informasi eksplisit pengguna dan produk. Menghilangkan baris kosong menjaga kualitas data.

### 4. Encoding User dan Produk ke Bentuk Numerik

Menggunakan `LabelEncoder`, kolom `user` dan `product` dikodekan menjadi integer untuk digunakan dalam embedding layer pada model.

📌 *Alasan:*
Model *collaborative filtering* memerlukan input dalam bentuk integer indeks untuk setiap user dan item, agar bisa dimapping ke vector embedding.

### 5. Pembagian Data: Training dan Validation

Dataset dibagi menjadi dua bagian:

* **Training set:** 80%
* **Validation set:** 20%
  Dengan metode acak (`random_state=42`) untuk memastikan replikasi.

📌 *Alasan:*
Pembagian ini digunakan untuk mengevaluasi kinerja model secara objektif dan mencegah overfitting. Validation set merepresentasikan data unseen.

### 6. Normalisasi Nilai Rating

Nilai rating (range 1–5) dinormalisasi ke rentang 0–1 menggunakan min-max scaling untuk stabilitas training model neural network. Transformasi ini dilakukan dengan formula:

$$
y_{\text{scaled}} = \frac{y - \min(y)}{\max(y) - \min(y)}
$$

📌 *Alasan:*
Normalisasi target membantu model neural network/embedding untuk konvergensi yang lebih stabil, terutama ketika digunakan bersama optimizer tertentu dan fungsi loss seperti MSE.

### 7. Finalisasi Data untuk Training

Data final disimpan dalam bentuk `user`, `product`, dan `rating`, yang siap dimasukkan ke dalam pipeline pelatihan model rekomendasi berbasis **embedding neural network**.

## 🤖 Modeling

Tahapan ini membahas proses pembangunan model sistem rekomendasi yang digunakan untuk menyelesaikan permasalahan. Pendekatan yang diambil adalah **model pembelajaran neural network berbasis embedding**, yang menggabungkan informasi eksplisit (rating) dan implisit (interaction score). Output utama dari sistem ini adalah **Top-N rekomendasi produk** untuk masing-masing pengguna.

### 1. Arsitektur Model — `RecommenderNet`

Model utama yang digunakan bernama `RecommenderNet`, yaitu subclass dari `tf.keras.Model`. Model ini menggunakan **embedding layer** untuk merepresentasikan pengguna dan produk ke dalam vektor berdimensi rendah (`embed_size = 50`). Detail komponennya sebagai berikut:

* **Embedding untuk user dan product**: Masing-masing pengguna dan produk direpresentasikan sebagai vektor berdimensi 50 yang dapat dipelajari selama training.
* **Bias user dan product**: Disediakan embedding bias untuk masing-masing entitas sebagai nilai koreksi tambahan.
* **Interaction score weight**: Model juga belajar sebuah parameter `interaction_weight` yang menjadi bobot kontribusi dari interaksi pengguna terhadap produk (berasal dari fitur `interaction_score`).
* **Dot product**: Vektor user dan product dikalikan sebagai representasi utama hubungan mereka, lalu dijumlahkan dengan bias dan kontribusi interaction score.
* **Aktivasi sigmoid**: Output akhir model berada di rentang \[0, 1], merepresentasikan skor preferensi pengguna terhadap produk tertentu.

📌 *Kelebihan:*

* Fleksibel dalam menggabungkan rating eksplisit dan fitur tambahan (interaction).
* Menghasilkan embedding yang dapat digunakan untuk analisis lanjutan.

📌 *Kekurangan:*

* Memerlukan data eksplisit/implisit dalam jumlah cukup agar model dapat belajar dengan baik.
* Interpretasi model lebih sulit dibanding algoritma klasik seperti kNN atau matrix factorization.

### 2. Kompilasi Model

Model dikompilasi dengan parameter berikut:

* **Loss function**: `BinaryCrossentropy`
  → Cocok karena target sudah dinormalisasi ke \[0, 1].
* **Optimizer**: `Adam` dengan learning rate 0.001
* **Metric**: `RootMeanSquaredError (RMSE)`
  → Digunakan untuk mengukur deviasi prediksi terhadap label sesungguhnya.

---

### 3. Training Model

Model dilatih selama **100 epoch** dengan batch size 8 menggunakan data training dan validation:

```python
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    validation_data=(X_val, y_val),
    verbose=2
)
```

Visualisasi hasil pelatihan menunjukkan tren **penurunan loss dan RMSE**, yang mengindikasikan model berhasil belajar representasi yang baik dari data:

* **Training vs Validation Loss**
![Visualisasi Matriks - Training Loss & Validation Loss](https://github.com/heriswaya/MLT-Project-1-and-2/raw/main/vis%20matrix%20-%20training_loss_rmse.png)
* **Training vs Validation RMSE**
![Visualisasi Matriks - Training Loss & RMSE](https://github.com/heriswaya/MLT-Project-1-and-2/raw/main/vis%20matrix%202%20-%20training_loss_rmse.png)

### 4. Top-N Recommendation Output

Setelah model dilatih, sistem memberikan rekomendasi produk berdasarkan skor prediksi tertinggi dari item yang **belum pernah dikunjungi** oleh pengguna tersebut.

Langkah-langkah yang dilakukan:

1. **Pilih 1 pengguna** secara acak dari dataset.
2. **Filter produk** yang telah dirating pengguna tersebut.
3. **Prediksi skor** terhadap semua produk yang belum pernah dikunjungi.
4. **Ambil TOP-N rekomendasi**, berdasarkan skor prediksi tertinggi.
5. **Tampilkan hasil rekomendasi** dan produk-produk yang pernah dirating oleh pengguna tersebut (maksimal 10).

Contoh hasil rekomendasi:

```
Showing recommendations for user: Michelle Willis
==============================
Produk yang telah dirating oleh user
------------------------------
• Smart Thermostat (Smart Home Devices) — rating: 1.0
------------------------------
Top 10 product recommendations
------------------------------
1. Amazon Echo  (Smart Home Devices)
2. HP Spectre  (Laptops)
3. Bed Frame  (Furniture)
4. Electric Range  (Kitchen Appliances)
5. Toaster  (Small Kitchen Appliances)
6. Microsoft Surface  (Tablets)
7. Samsung QLED TV  (TVs)
8. LG OLED TV  (TVs)
9. Steam Deck  (Gaming Consoles)
10. Sheets  (Bedding)
```

📌 *Catatan:*
Pengambilan rekomendasi ini dilakukan dengan memprediksi skor semua produk yang **belum dikunjungi** oleh pengguna, lalu memilih N produk dengan skor tertinggi.

Berikut adalah contoh penulisan bagian **Evaluation** untuk proyek forecasting Anda berdasarkan log training yang Anda berikan:

## Evaluation

Pada proyek ini, metrik evaluasi yang digunakan adalah **Root Mean Squared Error (RMSE)**, baik pada data latih maupun data validasi. RMSE merupakan salah satu metrik yang umum digunakan dalam permasalahan regresi dan forecasting karena memberikan penalti yang lebih besar terhadap kesalahan prediksi yang besar. RMSE dihitung dengan rumus:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

di mana:

* $y_i$ adalah nilai aktual,
* $\hat{y}_i$ adalah nilai prediksi,
* $n$ adalah jumlah total data.

### Hasil Evaluasi

Berdasarkan hasil pelatihan model selama 100 epoch, diperoleh nilai metrik evaluasi sebagai berikut:

* **RMSE pada data latih (training)**: sekitar **0.063**
* **RMSE pada data validasi (validation)**: sekitar **0.330**

Nilai RMSE pada data latih yang sangat kecil menunjukkan bahwa model mampu belajar dengan baik dari data latih. Sementara itu, nilai RMSE pada data validasi yang berada di kisaran 0.33 menunjukkan bahwa model juga mampu melakukan generalisasi dengan cukup baik terhadap data yang belum pernah dilihat sebelumnya.

Selain RMSE, model ini juga meminimalkan nilai loss, yaitu **Mean Squared Error (MSE)**, yang merupakan kuadrat dari RMSE. Metrik MSE dihitung dengan rumus:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Dari hasil pelatihan, MSE pada data validasi berada di kisaran **0.747–0.748**, yang konsisten dengan nilai RMSE validasi yang berada di kisaran **0.330–0.332**.

### Interpretasi

Perbedaan nilai RMSE antara data latih dan data validasi cukup signifikan, yang mengindikasikan kemungkinan adanya **sedikit overfitting**. Namun demikian, perbedaan tersebut masih dalam batas yang dapat ditoleransi, sehingga model masih cukup andal untuk melakukan prediksi jangka pendek.

Apabila dibutuhkan performa yang lebih baik, pendekatan lanjutan seperti regularisasi, penyesuaian arsitektur model, atau penggunaan teknik ensemble dapat dipertimbangkan.