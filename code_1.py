"""
Implementasi Konsep Deteksi Tepi dengan Operator Robert dan Sobel pada Gambar Rubber Duck.
"""

import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def robert_cross_operator(image_input):
    """Menerapkan operator Robert Cross untuk deteksi tepi."""
    filter_x = np.array([[-1, 0], [0, 1]])
    filter_y = np.array([[0, -1], [1, 0]])

    gradient_x = scipy.signal.convolve2d(image_input, filter_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image_input, filter_y, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

def sobel_operator(image_input):
    """Menerapkan operator Sobel untuk deteksi tepi."""
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = scipy.signal.convolve2d(image_input, filter_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image_input, filter_y, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

# Membaca gambar
IMAGE_PATH = 'assets/scenery.jpg'
image = np.asarray(iio.imread(IMAGE_PATH), dtype=np.float32)
if len(image.shape) == 3:  # Jika gambar RGB, konversi ke grayscale
    image = np.mean(image, axis=-1)

# Menerapkan operator Robert dan Sobel
robert_edges = robert_cross_operator(image)
sobel_edges = sobel_operator(image)

# Menampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')

plt.subplot(1, 3, 2)
plt.imshow(robert_edges, cmap='gray')
plt.title('Deteksi Tepi Robert')

plt.subplot(1, 3, 3)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Deteksi Tepi Sobel')

plt.tight_layout()
plt.show()

# Hasil Analisis
print("""
Operator Robert menghasilkan tepi lebih tipis dan sensitif terhadap perubahan kecil,
sedangkan Sobel menghasilkan tepi lebih tebal dan jelas karena mempertimbangkan lebih banyak piksel.

Secara keseluruhan, Sobel lebih baik dalam menangkap detail dan lebih tahan noise.
Robert lebih cocok untuk deteksi cepat, Sobel lebih cocok untuk hasil lebih akurat.
""")
