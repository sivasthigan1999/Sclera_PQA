import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets
from ultralytics import YOLO

################################################
# 1) HELPER FUNCTION FOR DISPLAY
################################################
def show_images_in_row(images, titles=None, cmap='gray', figsize=(15, 5)):
    """
    Display multiple images in a single row.
    """
    if titles is None:
        titles = ["" for _ in images]
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        # If single-channel, show in cmap='gray'
        if cmap and len(img.shape) == 2:
            plt.imshow(img, cmap=cmap)
        else:
            # If color (3-channel BGR), convert to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.show()

################################################
# 2) FUNCTION: Function to classify the IRIS Type
################################################
def classify_iris(image_path, threshold=100):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE to enhance contrast
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced_image = clahe.apply(image)
    
    # Calculate the mean intensity of the iris region
    mean_intensity = np.mean(image)
    # print(mean_intensity)
    
    # Classify based on threshold
    if mean_intensity > threshold:
        return "Light Iris"
    else:
        return "Dark Iris"

################################################
# 2) FUNCTION: Compute Grayscale Histogram
################################################
def compute_gray_hist(image_gray, mask=None, histSize=256, ranges=(0, 256)):
    """
    Compute a grayscale histogram for image_gray with an optional mask.
    Returns a normalized histogram (1D).
    """
    hist = cv2.calcHist([image_gray], [0], mask, [histSize], ranges)
    # Normalize histogram so that sum(hist) = 1 or use L2 norm
    hist_norm = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_norm

################################################
# 3) HELPER: Compute Boundary Sharpness
################################################
def compute_boundary_gradient(contour, gray_image, ksize=3):
    """
    For each point on the contour, sample its gradient magnitude in the grayscale image.
    Returns the average gradient magnitude along the contour boundary.
    """
    if len(contour) < 5:
        return 0.0

    # Compute gradient in the entire image first
    sobelx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=ksize)
    grad_mag = cv2.magnitude(sobelx, sobely)

    contour_pts = contour.reshape(-1, 2)
    values = []
    for (cx, cy) in contour_pts:
        if 0 <= cy < grad_mag.shape[0] and 0 <= cx < grad_mag.shape[1]:
            values.append(grad_mag[int(cy), int(cx)])
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))

################################################
# 4) OPTIONAL HELPER: Local Standard Deviation (Texture)
################################################
def local_std_of_region(gray_image, region_mask):
    """
    Computes the standard deviation of intensities within the region.
    A high value => more texture => likely iris.
    A low value => uniform => possibly glare.
    """
    coords = np.where(region_mask == 1)
    if len(coords[0]) == 0:
        return 0.0
    region_vals = gray_image[coords]
    return float(np.std(region_vals))

################################################
# 5) HELPER: Local Mean of Nearby Pixels
################################################
def local_mean_nearby(gray_image, contour, margin=20):
    """
    Computes the mean intensity in a local neighborhood
    around the bounding box of 'contour'.
      - margin: how many pixels to expand in each direction
    """
    x, y, w, h = cv2.boundingRect(contour)
    # Expand bounding box by 'margin'
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray_image.shape[1], x + w + margin)
    y2 = min(gray_image.shape[0], y + h + margin)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi))

################################################
# 6) POST-PROCESSING FUNCTION (WITH LOCAL OFFSET CHECK)
################################################
def postprocess_glare_mask(final_mask,
                           original_rgb,
                           global_hist,
                           # Existing parameters
                           min_area=10,
                           max_area=50000,
                           min_mean_v=50,
                           max_mean_s=200,
                           max_std_v=100,
                           max_edge_ratio=1.0,
                           hist_correlation_thr=0.9,
                           min_boundary_grad=10.0,
                           min_region_brightness=20,
                           max_local_std=200.0,
                           # Local mean-based offset
                           local_mean_margin=30,
                           local_offset=20):
    """
    Applies additional constraints to remove false positives, including
    a local neighborhood mean check with an offset:

    region_mean_intensity > (local_mean_of_area + local_offset) => pass as glare
    """
    # Convert to HSV for color checks
    hsv_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    refined_mask = np.zeros_like(final_mask)

    # Find connected components (contours)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # ---------------------------
        # 1) Area Check
        # ---------------------------
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Create mask just for this contour
        single_contour_mask = np.zeros_like(final_mask)
        cv2.drawContours(single_contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        region_indices = np.where(single_contour_mask == 1)

        # ---------------------------
        # 2) Boundary Sharpness
        # ---------------------------
        boundary_grad = compute_boundary_gradient(contour, gray_image, ksize=3)
        if boundary_grad < min_boundary_grad:
            continue

        # ---------------------------
        # 3) HSV brightness + saturation
        # ---------------------------
        v_values = hsv_image[..., 2][region_indices]
        s_values = hsv_image[..., 1][region_indices]
        mean_v = np.mean(v_values)
        mean_s = np.mean(s_values)

        if mean_v < min_mean_v:
            continue
        if mean_s > max_mean_s:
            continue

        # ---------------------------
        # 4) Uniformity check (std. dev. in V)
        # ---------------------------
        std_v = np.std(v_values)
        if std_v > max_std_v:
            continue

        # ---------------------------
        # 5) Edge ratio check
        # ---------------------------
        x, y, w, h = cv2.boundingRect(contour)
        sub_gray = gray_image[y:y+h, x:x+w]
        sub_mask = single_contour_mask[y:y+h, x:x+w]
        region_gray = cv2.bitwise_and(sub_gray, sub_gray, mask=(sub_mask*255).astype(np.uint8))

        edges = cv2.Canny(region_gray, threshold1=50, threshold2=150)
        total_pixels = np.count_nonzero(sub_mask)
        edge_count = np.count_nonzero(edges)
        edge_ratio = edge_count / float(total_pixels) if total_pixels > 0 else 0
        if edge_ratio > max_edge_ratio:
            continue

        # ---------------------------
        # 6) Distribution Similarity
        # ---------------------------
        region_hist = compute_gray_hist(region_gray, (sub_mask*255).astype(np.uint8), 256, (0,256))
        correlation = cv2.compareHist(global_hist, region_hist, cv2.HISTCMP_CORREL)
        if correlation > hist_correlation_thr:
            # If region is too similar to normal iris => skip
            continue

        # ---------------------------
        # 7) Region Brightness Check
        # ---------------------------
        region_pixel_values = gray_image[region_indices]
        max_intensity = np.max(region_pixel_values)
        mean_intensity = np.mean(region_pixel_values)
        if max_intensity < min_region_brightness and mean_intensity < min_region_brightness:
            continue

        # ---------------------------
        # 8) Optional Local Texture Check
        # ---------------------------
        region_local_std = local_std_of_region(gray_image, single_contour_mask)
        if region_local_std > max_local_std:
            continue

        # ---------------------------
        # 9) Local Mean + Offset Check
        #    region_mean must exceed (local_mean + offset)
        # ---------------------------
        local_mean_val = local_mean_nearby(gray_image, contour, margin=local_mean_margin)
        if mean_intensity <= (local_mean_val + local_offset):
            # Not significantly brighter than its neighborhood => skip
            continue

        # If all checks pass => keep
        cv2.drawContours(refined_mask, [contour], -1, 1, thickness=cv2.FILLED)

    return refined_mask

################################################
# 7) YOLO SEGMENTATION + PIPELINE
################################################
def analyze_glare(gray_sclera):
    
    # 3) Wavelet detection
    gray_orig = gray_sclera
    gray_bgr = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
    cropped_rgb=gray_bgr
    wavelet_name = 'haar'
    coeffs2 = pywt.dwt2(gray_orig, wavelet=wavelet_name)
    LL, (LH, HL, HH) = coeffs2

    def threshold_subband(subband, thr):
        return (np.abs(subband) > thr).astype(np.uint8)

    wavelet_thr = 10
    mask_LH = threshold_subband(LH, wavelet_thr)
    mask_HL = threshold_subband(HL, wavelet_thr)
    mask_HH = threshold_subband(HH, wavelet_thr)

    wavelet_subband_mask = np.logical_or.reduce([mask_LH, mask_HL, mask_HH]).astype(np.uint8)

    subband_h, subband_w = LH.shape
    orig_h, orig_w = gray_orig.shape
    wavelet_mask = cv2.resize(wavelet_subband_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    wavelet_mask = cv2.morphologyEx(wavelet_mask, cv2.MORPH_OPEN, kernel_open)
    wavelet_mask = cv2.morphologyEx(wavelet_mask, cv2.MORPH_CLOSE, kernel_close)

    cnts, _ = cv2.findContours(wavelet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wavelet_final = np.zeros_like(wavelet_mask)
    cv2.drawContours(wavelet_final, cnts, -1, color=1, thickness=cv2.FILLED)

    show_images_in_row(
        [gray_orig, wavelet_mask*255, wavelet_final*255],
        ["Gray Cropped", "Wavelet Mask (Morph)", "Wavelet Mask (Filled)"]
    )

    # 4) Inpaint + local threshold
    black_mask = np.all(cropped_rgb == [0,0,0], axis=-1).astype(np.uint8)
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(cropped_bgr, black_mask, 1, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    gray_inpainted = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2GRAY)
    blur_ksize = 31
    offset = 15
    blur_img = cv2.GaussianBlur(gray_inpainted, (blur_ksize, blur_ksize), 0)
    local_mask = (gray_inpainted.astype(np.int32) - blur_img.astype(np.int32)) > offset
    local_mask = local_mask.astype(np.uint8)

    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel_open)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel_close)

    cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    local_filled = np.zeros_like(local_mask)
    cv2.drawContours(local_filled, cnts, -1, color=1, thickness=cv2.FILLED)

    show_images_in_row(
        [gray_inpainted, blur_img, local_filled*255],
        ["Inpainted Gray", "Blurred Img", "Local Filled"]
    )

    # 5) Combine Wavelet + Local
    combined_mask = cv2.bitwise_or(wavelet_final, local_filled)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined_mask)
    cv2.drawContours(final_mask, cnts, -1, color=1, thickness=cv2.FILLED)

    show_images_in_row(
        [wavelet_final*255, local_filled*255, final_mask*255],
        ["Wavelet Final", "Local Final", "Combined Mask"]
    )

    # 6) Create Global Hist of Iris
    global_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
    iris_mask = (black_mask == 0).astype(np.uint8)
    iris_hist = compute_gray_hist(global_gray, iris_mask*255, histSize=256, ranges=(0,256))

    # 7) Post-Processing (Offset-based local mean check)
    refined_mask = postprocess_glare_mask(
        final_mask,
        cropped_rgb,
        iris_hist,
        min_area=10,
        max_area=50000,
        min_mean_v=50,
        max_mean_s=200,
        max_std_v=100,
        max_edge_ratio=0.9,
        hist_correlation_thr=0.7,
        min_boundary_grad=60.0,
        min_region_brightness=80,
        max_local_std=100.0,
        # New local mean-based offset
        local_mean_margin=50, # how large a neighborhood around the contour bounding box
        local_offset=20        # region mean must exceed local_mean + 20
    )

    show_images_in_row(
        [final_mask*255, refined_mask*255],
        ["Mask Before Post-Process", "Refined Mask After Post-Process"]
    )

    # 8) Visualize on Cropped
    glare_only = cv2.bitwise_and(cropped_rgb, cropped_rgb, mask=refined_mask)
    overlay = cropped_rgb.copy()
    overlay[refined_mask == 1] = (255, 0, 0)
    alpha = 0.5
    glare_overlay = cv2.addWeighted(overlay, alpha, cropped_rgb, 1 - alpha, 0)

    show_images_in_row(
        [cropped_rgb, glare_only, glare_overlay],
        ["Cropped (RGB)", "Glare Regions Only", "Glare Highlighted"],
        cmap=None
    )

    # 9) Calculate glare percentage
    # Count non-black pixels in cropped_rgb (sclera region)
    sclera_pixels = np.sum(iris_mask)  # Count pixels where black_mask is False
    
    # Count glare pixels from refined mask
    glare_pixels = np.sum(refined_mask)
    
    # Calculate percentage
    glare_percentage = (glare_pixels / sclera_pixels) * 100 if sclera_pixels > 0 else 0
    
    print(f"Glare percentage in sclera: {glare_percentage:.2f}%")
    return glare_percentage
