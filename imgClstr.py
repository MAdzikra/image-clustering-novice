import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import os

# Fungsi untuk menggabungkan pixel dari beberapa gambar
def gather_pixels_from_images(images):
    all_pixels = []
    for image in images:
        image_array = np.array(image)
        pixel_values = image_array.reshape((-1, 3))  # Flatten image to a list of pixels
        all_pixels.append(pixel_values)
    return np.vstack(all_pixels)  # Gabungkan semua pixel dari semua gambar

# Fungsi untuk melakukan sampling pada data
def sample_data_for_silhouette(all_pixels, labels, sample_size=10000):
    # Jika ukuran dataset lebih besar dari sample_size, ambil sampel
    if len(all_pixels) > sample_size:
        sampled_pixels, sampled_labels = resample(all_pixels, labels, n_samples=sample_size, random_state=42)
        return sampled_pixels, sampled_labels
    # Jika dataset lebih kecil dari sample_size, gunakan seluruh data
    return all_pixels, labels

# Fungsi untuk melakukan clustering pada semua gambar sekaligus
def cluster_images(images, n_clusters, max_iters=100, sample_size=10000):
    # Gabungkan semua pixel dari gambar yang digunakan
    all_pixels = gather_pixels_from_images(images)
    all_pixels = np.float32(all_pixels)

    # Inisialisasi centroid secara acak dari semua pixel gabungan
    np.random.seed(42)
    centroids = all_pixels[np.random.choice(all_pixels.shape[0], n_clusters, replace=False)]

    # Iterasi untuk mengoptimalkan posisi centroid
    for _ in range(max_iters):
        distances = cdist(all_pixels, centroids, metric='euclidean')  # Hitung jarak dari semua pixel ke centroid
        labels = np.argmin(distances, axis=1)  # Tentukan label untuk pixel terdekat

        # Hitung centroid baru
        new_centroids = np.array([all_pixels[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Jika centroid tidak berubah, hentikan iterasi
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Lakukan sampling pada data untuk menghitung silhouette score
    sampled_pixels, sampled_labels = sample_data_for_silhouette(all_pixels, labels, sample_size)
    
    # Menghitung silhouette score pada subset data
    silhouette_avg = silhouette_score(sampled_pixels, sampled_labels)

    return centroids, labels, silhouette_avg

# Fungsi untuk mengaplikasikan centroid yang dihitung ke masing-masing gambar
def apply_centroids_to_image(image, centroids):
    image_array = np.array(image)
    pixel_values = image_array.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Hitung jarak dari pixel ke setiap centroid
    distances = cdist(pixel_values, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    # Buat gambar tersegmentasi berdasarkan label cluster
    segmented_image = centroids[labels]
    segmented_image = segmented_image.reshape(image_array.shape)

    return np.uint8(segmented_image)

# Fungsi untuk menampilkan gambar asli dan hasil clustering
def show_image_clustering(original_image, clustered_image, n_clusters):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Gambar Asli")
    axes[0].axis("off")

    axes[1].imshow(clustered_image)
    axes[1].set_title(f"Clustering dengan {n_clusters} Cluster")
    axes[1].axis("off")

    st.pyplot(fig)

# Fungsi utama Streamlit
def main():
    st.title("KMeans Image Clustering dengan Sampling untuk Silhouette Score")
    st.text("Oleh: ")
    st.text("- Candra Wibawa (140810220044)")
    st.text("- Muhammad Adzikra Dhiya Alfauzan (140810220046)")
    st.text("- Ivan Arsy Himawan (140810220052)")

    # Upload gambar dari user
    uploaded_files = st.file_uploader("Upload gambar (multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Slider untuk memilih jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=5, value=3)

    if uploaded_files:
        # Membaca gambar yang diupload
        images = [Image.open(file) for file in uploaded_files]

        # Lakukan clustering pada dataset gabungan
        centroids, _, silhouette_avg = cluster_images(images, n_clusters)

        st.write(f"Silhouette Coefficient untuk {n_clusters} clusters (dengan sampling): {silhouette_avg:.4f}")

        # Tampilkan clustering untuk setiap gambar yang diupload
        for image in images:
            clustered_image = apply_centroids_to_image(image, centroids)
            show_image_clustering(image, clustered_image, n_clusters)

    else:
        st.warning("Silakan upload minimal satu gambar untuk di-cluster.")

if __name__ == '__main__':
    main()
