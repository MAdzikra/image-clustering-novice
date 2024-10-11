import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

# Fungsi untuk melakukan clustering pada gambar menggunakan algoritma K-Means manual dengan optimisasi
def cluster_image_manual_fast(image, n_clusters, max_iters=100):
    image_array = np.array(image)
    pixel_values = image_array.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Inisialisasi centroid secara acak
    np.random.seed(42)
    centroids = pixel_values[np.random.choice(pixel_values.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):
        # Menghitung jarak semua pixel ke setiap centroid (menggunakan vektorisasi cdist)
        distances = cdist(pixel_values, centroids, metric='euclidean')
        
        # Assign setiap pixel ke centroid terdekat
        labels = np.argmin(distances, axis=1)

        # Hitung centroid baru
        new_centroids = np.array([pixel_values[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Jika centroid tidak berubah, hentikan
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Buat gambar tersegmentasi berdasarkan label cluster
    segmented_image = centroids[labels]
    segmented_image = segmented_image.reshape(image_array.shape)

    return np.uint8(segmented_image), labels.reshape(image_array.shape[:2])

# Fungsi untuk menghitung pusat dari tiap cluster
def get_cluster_centers(labels, n_clusters):
    cluster_centers = []
    
    for i in range(n_clusters):
        cluster_pixels = np.argwhere(labels == i)
        if len(cluster_pixels) > 0:
            # Ambil rata-rata posisi pixel untuk menghitung pusat cluster
            center_y, center_x = cluster_pixels.mean(axis=0).astype(int)
            cluster_centers.append((center_x, center_y))
    
    return cluster_centers

# Fungsi untuk menambahkan label angka di atas gambar
def add_labels_to_image(clustered_image, cluster_centers):
    # Convert gambar ke format yang bisa diolah oleh OpenCV
    clustered_image_with_labels = clustered_image.copy()

    # Tambahkan angka label ke pusat tiap cluster
    for i, (center_x, center_y) in enumerate(cluster_centers):
        # Tambahkan teks label ke gambar (menggunakan cv2.putText)
        cv2.putText(clustered_image_with_labels, str(i), (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return clustered_image_with_labels

# Fungsi untuk menampilkan gambar asli dan hasil clustering dengan label angka di tiap cluster
def show_image_clustering_with_labels(original_image, clustered_image, labels, n_clusters):
    # Dapatkan pusat dari tiap cluster
    cluster_centers = get_cluster_centers(labels, n_clusters)

    # Tambahkan label angka di atas gambar clustering
    clustered_image_with_labels = add_labels_to_image(clustered_image, cluster_centers)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Gambar asli
    axes[0].imshow(original_image)
    axes[0].set_title("Gambar Asli")
    axes[0].axis("off")

    # Gambar clustering dengan label angka
    axes[1].imshow(clustered_image_with_labels)
    axes[1].set_title(f"Clustering dengan {n_clusters} Cluster")
    axes[1].axis("off")

    st.pyplot(fig)

# Fungsi utama Streamlit
def main():
    st.title("KMeans Image Clustering (Manual) dengan Optimisasi dan Label Cluster")

    # Slider untuk memilih jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=5, value=3)
    
    # Upload gambar dari user
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    # Jika user tidak mengupload gambar, tampilkan gambar default
    if uploaded_file is None:
        st.subheader("Contoh Gambar")
        
        # Nama file gambar default
        images = ['001.jpg', '002.jpg', '003.jpg', '004.jpg', '005.jpg']
        
        # Tampilkan gambar default dengan clustering
        for image_path in images:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                
                # Lakukan clustering pada gambar
                clustered_image, labels = cluster_image_manual_fast(image, n_clusters)
                
                # Tampilkan gambar asli dan hasil clustering dengan label
                show_image_clustering_with_labels(image, clustered_image, labels, n_clusters)
            else:
                st.warning(f"Gambar {image_path} tidak ditemukan di folder.")

    # Jika user mengupload gambar, tampilkan gambar tersebut
    if uploaded_file is not None:
        st.subheader("Gambar yang Diupload")

        # Membaca gambar yang diupload
        image = Image.open(uploaded_file)

        # Tampilkan gambar asli
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        # Tampilkan hasil clustering dengan label angka
        clustered_image, labels = cluster_image_manual_fast(image, n_clusters)
        show_image_clustering_with_labels(image, clustered_image, labels, n_clusters)

if __name__ == '__main__':
    main()
