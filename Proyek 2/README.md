# Laporan Proyek Machine Learning - Christopher Adrian Kusuma

## Project Overview

Membaca buku merupakan aspek yang sangat penting untuk membentuk sifat seseorang, mengembangkan metode berpikir dan menciptakan ide baru. Orang yang sering membaca buku cenderung memiliki wawasan yang luas dan memiliki kemampuan belajar yang tinggi [1].

Melihat pentingnya manfaat membaca buku, kita perlu membaca buku lebih banyak lagi. Kita tentu saja memiliki beberapa preferensi saat membaca buku. Selain itu, ada banyak sekali buku yang diterbitkan dan perlu kita saring sesuai dengan preferensi kita. Oleh karena itu, dibutuhkan sistem rekomendasi yang akan menyaring buku-buku tersebut untuk kita.

## Business Understanding

### Problem Statements

Adapun permasalahan yang diangkat pada proyek ini adalah:
- Bagaimana cara memilih buku yang sesuai dengan preferensi seseorang?

### Goals

Adapun tujuan proyek ini adalah:
- Mengetahui cara memilih buku yang sesuai dengan preferensi seseorang

### Solution statements
Menggunakan 2 pendekatan berbeda untuk sistem rekomendasi, yaitu:
1. Content-based
  
    Algoritma yang digunakan pada pendekatan ini adalah penghitungan kemiripan sebuah buku berdasarkan penulisnya.

2. Collaborative Filtering

    Algoritma yang digunakan pada pendekatan ini adalah menghitung kemungkinan rating yang akan diberikan pengguna menggunakan deep learning.

## Data Understanding
Dataset diunduh dari [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini terdiri dari 3 file csv:
- Books.csv
- Ratings.csv
- Users.csv

![](images/bt_1.png)

Variabel-variabel pada Books.csv adalah sebagai berikut:
- ISBN: ISBN dari buku
- Book-Title: Judul buku
- Book-Author: Penulis buku
- Year-Of-Publication: Tahun terbit buku
- Publisher: Penerbit buku
- Image-URL-S: URL gambar buku ukuran S
- Image-URL-M: URL gambar buku ukuran M
- Image-URL-L: URL gambar buku ukuran L

![](images/rt_1.png)

Variabel-variabel pada Ratings.csv adalah sebagai berikut:
- User-ID: ID pengguna
- ISBN: ISBN dari buku
- Book-Rating: Rating buku

![](images/ut_1.png)

Variabel-variabel pada Users.csv adalah sebagai berikut:
- User-ID: ID pengguna
- Location: Lokasi pengguna
- Age: Umur pengguna

Analisis data:
### Books
![](images/bt_2.png)

Terdapat beberapa data yang memiliki format yang salah dan bisa diperbaiki dengan mudah

![](images/bt_3.png)

![book_1](https://user-images.githubusercontent.com/32239110/183849392-88c1b066-f5d2-46c5-bd17-3f64a7e9a37d.png)

Melihat gambar tersebut, dapat dilihat bahwa tahun terbit buku bervariasi dari tahun 0 hingga 2050. Akan tetapi, buku yang terbit di tahun 0 kemungkinan merupakan kesalahan pada data, begitu pula dengan buku yang terbit di tahun 2050. Dari grafik, dapat diketahui bahwa buku kebanyakan diterbitkan pada tahun 1970 hingga 2010. Karena itu, penulis melakukan filtering buku hanya untuk yang terbit pada tahun 1970 hingga 2010 dengan persebaran ditunjukkan gambar berikut

![book_2](https://user-images.githubusercontent.com/32239110/183849432-9c23e26a-946f-4586-8fbc-618ef16e4a47.png)

### Ratings
![rating_1](https://user-images.githubusercontent.com/32239110/183849481-466ce11f-83fb-4f92-a58c-1efa247ba214.png)

![](images/rt_2.png)

Melihat grafik tersebut, sangat banyak buku yang memiliki rating 0. Dari tabel dapat terlihat bahwa terdapat beberapa buku terkenal yang mendapat rating 0 dari cukup banyak pengguna. Ada 2 kemungkinan untuk hal tersebut, yaitu rating tersebut merupakan nilai default karena pengguna belum memberikan rating untuk buku tersebut atau memang pengguna memberikan rating 0 terhadap buku tersebut.

### Users
![age_1](https://user-images.githubusercontent.com/32239110/183849514-a55c0fe1-b1f1-486a-b56d-615d247dde78.png)

Dapat dilihat persebaran umur pengguna yang beragam dari 0 hingga 244. Dari gambar ini dapat dilihat bahwa pembaca buku kebanyakan berasal dari golongan usia anak-anak hingga dewasa.

## Data Preparation

### Books
Penulis menghapus kolom Image-URL-S, Image-URL-M, dan Image-URL-L karena tidak diperlukan. Selain itu, tidak terdapat null values pada dataset yang digunakan.
```python
books_df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)
```

### Users
Terdapat banyak null values pada tabel users. Selain itu, usia pengguna bervariasi dari 0 hingga 244. Sehingga penulis hanya mengambil user yang berumur antara 10 hingga 70 tahun.
```python
users_df = users_df[(users_df['Age'] >= 10) & (users_df['Age'] <= 70)]
users_df.dropna(inplace=True)
```

### Merging

![](images/merge.png)

Penulis menggabungkan ketiga dataset untuk digunakan pada tahap modelling.

## Modeling
Pemodelan dilakukan dengan menggunakan 2 metode yang berbeda. Masing-masing metode memberikan n recommendation teratas untuk pengguna.

### Content-based Filtering
Metode ini dilakukan dengan menghitung kemiripan buku berdasarkan penulisnya. Penulis mengimplementasikan algoritma ini menggunakan Tf-Idf dan menghitung kemiripan buku menggunakan cosine similarity. Rekomendasi buku didasarkan pada kemiripan buku dengan buku yang pernah dibaca pengguna tersebut sebelumnya.
```python
vectorizer = TfidfVectorizer()
vectorizer.fit(book_author_df['Book-Author'])
tfidf_matrix = vectorizer.transform(book_author_df['Book-Author'])
cosine_sim = cosine_similarity(tfidf_matrix) 

def book_recommendations(book_name, similarity_data=cosine_sim_df, items=book_author_df[['Book-Title', 'Book-Author']], k=5):
  index = similarity_data.loc[:, book_name].to_numpy().argpartition(range(-1, -k, -1))
  
  closest = similarity_data.columns[index[-1:-(k+2):-1]]
  closest = closest.drop(book_name, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)
```
Berikut top n recommendations yang diberikan:
![content](https://user-images.githubusercontent.com/32239110/183846583-f494983e-a669-4ab6-9626-323eaa3daf7b.png)

### Collaborative Filtering
Metode ini merupakan metode filtering dengan membandingkan antar pengguna. Penulis mengimplementasikan algoritma ini menggunakan deep learning dengan aktivasi sigmoid. Rekomendasi buku didasarkan pada kemungkinan rating yang akan diberikan pengguna terhadap buku yang belum pernah dibaca.
```
rating = sigmoid((user_vector . book_vector) + user_bias + book_bias)
```
Berikut top n recommendations yang diberikan:
![colab](https://user-images.githubusercontent.com/32239110/183846670-e52f6225-c519-4665-bf97-534d46f33154.png)

## Evaluation

### Content-based Filtering
Melihat rekomendasi yang diberikan, sistem cenderung merekomendasikan buku dari penulis yang memiliki nama yang mirip (Wendy atau Hobson). Hal ini menandakan sistem rekomendasi sudah dapat memberikan rekomendasi berdasarkan penulis dari buku yang pernah dibaca oleh pengguna.

### Collaborative Filtering
Model dibuat dengan aktivasi sigmoid dengan loss function mse dan mendapat val_loss sebesar 0,0736.

![mse](https://user-images.githubusercontent.com/32239110/183843459-0ac045f3-ce08-4578-8e61-dc254f9393c4.png)

![loss](https://user-images.githubusercontent.com/32239110/183843519-0a7f3a3f-5849-4e64-ba2e-2dd563d596d3.png)

Sistem bisa memberikan rekomendasi buku yang kemungkinan akan mendapat rating yang tinggi dari pengguna.

## Referensi:
1. [Reading Habits Among Students and its Effect on Academic Performance: A Study of Students of Koforidua Polytechnic](https://www.academia.edu/download/52948426/fulltext.pdf)
