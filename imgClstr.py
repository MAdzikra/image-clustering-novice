import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

# Fungsi untuk menggabungkan pixel dari beberapa gambar
def gather_pixels_from_images(images):
    all_pixels = []
    for image in images:
        image_array = np.array(image)
        pixel_values = image_array.reshape((-1, 3))  # Flatten image to a list of pixels
        all_pixels.append(pixel_values)
    return np.vstack(all_pixels)  # Gabungkan semua pixel dari semua gambar

# Fungsi untuk melakukan clustering pada semua gambar sekaligus
def cluster_images(images, n_clusters, max_iters=100):
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

    return centroids, labels

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

    return np.uint8(segmented_image), labels.reshape(image_array.shape[:2])

# Fungsi untuk menambahkan label angka di atas gambar
def add_labels_to_image(clustered_image, cluster_centers):
    clustered_image_with_labels = clustered_image.copy()

    # Tambahkan angka label ke pusat tiap cluster
    for i, (center_x, center_y) in enumerate(cluster_centers):
        cv2.putText(clustered_image_with_labels, str(i), (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return clustered_image_with_labels

# Fungsi untuk menghitung pusat dari tiap cluster
def get_cluster_centers(labels, n_clusters):
    cluster_centers = []
    for i in range(n_clusters):
        # Ambil semua pixel yang termasuk dalam cluster i
        cluster_pixels = np.argwhere(labels == i)
        if len(cluster_pixels) > 0:
            # Hitung rata-rata posisi pixel untuk menentukan pusat cluster
            center_y, center_x = cluster_pixels.mean(axis=0).astype(int)
            cluster_centers.append((center_x, center_y))
    return cluster_centers


# Fungsi untuk menampilkan gambar asli dan hasil clustering dengan label angka di tiap cluster
def show_image_clustering_with_labels(original_image, clustered_image, labels, n_clusters):
    cluster_centers = get_cluster_centers(labels, n_clusters)
    clustered_image_with_labels = add_labels_to_image(clustered_image, cluster_centers)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Gambar Asli")
    axes[0].axis("off")

    axes[1].imshow(clustered_image_with_labels)
    axes[1].set_title(f"Clustering dengan {n_clusters} Cluster")
    axes[1].axis("off")

    st.pyplot(fig)

# Fungsi utama Streamlit
def main():
    st.title("KMeans Image Clustering Konsisten untuk Semua Gambar")
    st.text("Oleh: ")


    # Upload gambar dari user
    uploaded_files = st.file_uploader("Upload gambar (multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Slider untuk memilih jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=5, value=3)

    if uploaded_files:
        # Membaca gambar yang diupload
        images = [Image.open(file) for file in uploaded_files]

        # Lakukan clustering pada dataset gabungan
        centroids, _ = cluster_images(images, n_clusters)

        # Tampilkan clustering untuk setiap gambar yang diupload
        for image in images:
            clustered_image, labels = apply_centroids_to_image(image, centroids)
            show_image_clustering_with_labels(image, clustered_image, labels, n_clusters)

    else:
        st.warning("Silakan upload minimal satu gambar untuk di-cluster.")

if __name__ == '__main__':
    main()
