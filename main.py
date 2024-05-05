import streamlit as st
import cv2
import numpy as np


# Function to apply Lowpass Gaussian filter
def apply_gaussian_filter(image):
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    diff = np.abs(image - gaussian_filtered)
    return gaussian_filtered, diff


# Function to apply Lowpass Butterworth filter
def apply_butterworth_filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a Butterworth filter
    r = 30
    n = 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = 1 / (1 + ((i - crow) ** 2 + (j - ccol) ** 2) / (r ** 2)) ** (2 * n)

    # Apply filter in the frequency domain
    fshift = np.fft.fft2(image)
    fshift = fshift * (1 - mask)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image to be in the range [0, 1]
    img_back = img_back / np.max(img_back)

    diff = np.abs(image - img_back)
    return img_back, diff


# Function to apply HighPass Laplacian filter
def apply_laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Normalize the image to be in the range [0, 1]
    laplacian = (laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian))

    diff = np.abs(image - laplacian)
    return laplacian, diff


# Function to perform Histogram Matching
def histogram_matching(source, target):
    if target is None:
        return source, np.zeros_like(source)

    matched = np.zeros_like(source)
    for i in range(source.shape[2]):
        matched[:, :, i] = cv2.equalizeHist(source[:, :, i], target[:, :, i])
    diff = np.abs(source - matched)
    return matched, diff


# Function to display image
def show_image(image, title):
    st.image(image, caption=title, use_column_width=True)


# Function to display difference image
def show_difference(diff, title):
    # Normalize the difference image to be in the range [0, 1]
    diff_normalized = diff / np.max(diff)
    st.image(diff_normalized, caption=title, use_column_width=True)


# Main function
def main():
    st.title("Image Filters App🥰")
    st.sidebar.title("Filters")

    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        show_image(image, "Original Image")

        # Checkboxes for filters
        apply_gaussian = st.sidebar.checkbox("Lowpass Gaussian Filter")
        apply_butterworth = st.sidebar.checkbox("Lowpass Butterworth Filter")
        apply_laplacian = st.sidebar.checkbox("HighPass Laplacian Filter")
        apply_hist_match = st.sidebar.checkbox("Histogram Matching")

        if apply_gaussian:
            gaussian_filtered, diff = apply_gaussian_filter(image_gray)
            show_image(gaussian_filtered, "Gaussian Filtered")
            show_difference(diff, "Difference")

        if apply_butterworth:
            butterworth_filtered, diff = apply_butterworth_filter(image_gray)
            show_image(butterworth_filtered, "Butterworth Filtered")
            show_difference(diff, "Difference")

        if apply_laplacian:
            laplacian_filtered, diff = apply_laplacian_filter(image_gray)
            show_image(laplacian_filtered, "Laplacian Filtered")
            show_difference(diff, "Difference")

        if apply_hist_match:
            target_image = cv2.imread('target_image.jpg')
            hist_matched, diff = histogram_matching(image, target_image)
            show_image(hist_matched, "Histogram Matched")
            show_difference(diff, "Difference")


if __name__ == '__main__':
    main()
