import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import os

# Fungsi untuk melakukan clustering pada gambar
def cluster_image(image, n_clusters):
    image_array = np.array(image)
    pixel_values = image_array.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)

    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_array.shape)
    
    return segmented_image

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
    st.title("KMeans Image Clustering")

    # Slider untuk memilih jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=3)
    
    # Upload gambar dari user
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    # Jika user tidak mengupload gambar, tampilkan gambar default
    if uploaded_file is None:
        st.subheader("Gambar Default")
        
        # Nama file gambar default
        images = ['001.jpg', '002.jpg', '003.jpg', '004.jpg', '005.jpg']
        
        # Tampilkan gambar default dengan clustering
        for image_path in images:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                
                # Lakukan clustering pada gambar
                clustered_image = cluster_image(image, n_clusters)
                
                # Tampilkan gambar asli dan hasil clustering
                show_image_clustering(image, clustered_image, n_clusters)
            else:
                st.warning(f"Gambar {image_path} tidak ditemukan di folder.")

    # Jika user mengupload gambar, tampilkan gambar tersebut
    if uploaded_file is not None:
        st.subheader("Gambar yang Diupload")

        # Membaca gambar yang diupload
        image = Image.open(uploaded_file)

        # Tampilkan gambar asli
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        # Tampilkan hasil clustering
        clustered_image = cluster_image(image, n_clusters)
        show_image_clustering(image, clustered_image, n_clusters)

if __name__ == '__main__':
    main()
