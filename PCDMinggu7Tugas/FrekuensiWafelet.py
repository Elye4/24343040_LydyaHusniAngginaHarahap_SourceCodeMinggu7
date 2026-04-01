import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import time

# =========================
# 1. LOAD CITRA
# =========================
img = cv2.imread('citra_natural.jpg', 0)
img = cv2.resize(img, (256,256))

# =========================
# 2. BUAT NOISE PERIODIK
# =========================
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

noise = 30 * np.sin(2*np.pi*X/20)
img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)

# =========================
# 3. FFT ANALYSIS
# =========================
def fft_analysis(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(1 + np.abs(fshift))
    phase = np.angle(fshift)

    return f, fshift, magnitude, phase

# =========================
# 4. REKONSTRUKSI
# =========================
def reconstruct(magnitude, phase):
    complex_img = np.exp(magnitude) * np.exp(1j*phase)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(complex_img)))
    return np.clip(img_back, 0, 255)

# =========================
# 5. FILTER
# =========================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    crow, ccol = rows//2, cols//2

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) < cutoff:
                mask[i,j] = 1
    return mask

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            d = (i-crow)**2 + (j-ccol)**2
            mask[i,j] = np.exp(-d/(2*(cutoff**2)))
    return mask

def notch_filter(shape):
    mask = np.ones(shape)
    crow, ccol = shape[0]//2, shape[1]//2

    mask[crow-10:crow+10, ccol-30:ccol-20] = 0
    mask[crow-10:crow+10, ccol+20:ccol+30] = 0

    return mask

# =========================
# 6. APPLY FILTER
# =========================
def apply_filter(image, mask):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

    return np.clip(img_back, 0, 255)

# =========================
# 7. WAVELET
# =========================
def wavelet_process(image):
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    recon = pywt.waverec2(coeffs, 'haar')
    return coeffs, np.clip(recon, 0, 255)

# =========================
# 8. PSNR
# =========================
def psnr(original, test):
    mse = np.mean((original - test)**2)
    if mse == 0:
        return 100
    return 10 * np.log10(255**2 / mse)

# =========================
# 9. PROSES UTAMA
# =========================
start = time.time()

# FFT
_, fshift, mag, phase = fft_analysis(img)

# Rekonstruksi
recon_phase = reconstruct(np.ones_like(mag), phase)
recon_mag = reconstruct(mag, np.zeros_like(phase))

# Filter
ideal = ideal_lowpass(img.shape, 30)
gauss = gaussian_lowpass(img.shape, 30)
notch = notch_filter(img.shape)

ideal_img = apply_filter(img_noise, ideal)
gauss_img = apply_filter(img_noise, gauss)
notch_img = apply_filter(img_noise, notch)

# Wavelet
coeffs, wavelet_img = wavelet_process(img_noise)

end = time.time()

# =========================
# 10. OUTPUT METRIK
# =========================
print("PSNR Ideal:", psnr(img, ideal_img))
print("PSNR Gaussian:", psnr(img, gauss_img))
print("PSNR Notch:", psnr(img, notch_img))
print("PSNR Wavelet:", psnr(img, wavelet_img))
print("Waktu:", end - start)

# =========================
# 11. VISUALISASI
# =========================
plt.figure(figsize=(12,8))

plt.subplot(3,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(3,3,2)
plt.imshow(img_noise, cmap='gray')
plt.title("Noise Periodik")

plt.subplot(3,3,3)
plt.imshow(mag, cmap='gray')
plt.title("Magnitude")

plt.subplot(3,3,4)
plt.imshow(ideal_img, cmap='gray')
plt.title("Ideal LPF")

plt.subplot(3,3,5)
plt.imshow(gauss_img, cmap='gray')
plt.title("Gaussian LPF")

plt.subplot(3,3,6)
plt.imshow(notch_img, cmap='gray')
plt.title("Notch Filter")

plt.subplot(3,3,7)
plt.imshow(recon_phase, cmap='gray')
plt.title("Phase Only")

plt.subplot(3,3,8)
plt.imshow(recon_mag, cmap='gray')
plt.title("Magnitude Only")

plt.subplot(3,3,9)
plt.imshow(wavelet_img, cmap='gray')
plt.title("Wavelet")

plt.tight_layout()
plt.show()