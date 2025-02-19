"""
Implementasi Segmentasi Citra Berbasis Clustering Menggunakan K-Means.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Membaca gambar
import imageio.v2 as iio
image = iio.imread('assets/scenery.jpg')

# Mengubah gambar menjadi array 2D (reshape)
pixels = image.reshape((-1, 3))

# Menentukan jumlah kluster (K)
K = 4  # Bisa diubah sesuai kebutuhan
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(pixels)

# Mengubah setiap piksel menjadi warna klusternya
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)

# Menampilkan hasil segmentasi
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f"Segmentasi K-Means (K={K})")
plt.axis("off")

plt.show()
