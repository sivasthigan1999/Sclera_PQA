import cv2
import numpy as np
import matplotlib.pyplot as plt
def analyze_sclera_exposure(gray_sclera, sclera_region, sclera_mask,plot_results=True):

    
    # Convert sclera region to grayscale if it's not already
    gray_sclera = cv2.cvtColor(sclera_region, cv2.COLOR_BGR2GRAY) if len(sclera_region.shape) == 3 else sclera_region
    
    # Normalize grayscale image to enhance contrast
    normalized_sclera = cv2.normalize(gray_sclera, None, 0, 255, cv2.NORM_MINMAX)

    # Compute histogram only for the sclera pixels
    sclera_pixels = gray_sclera[sclera_mask > 0]
    
    if len(sclera_pixels) == 0:
        return {
            'status': 'error',
            'message': 'Eye or iris mask not detected properly!',
            'mean_brightness': None,
            'exposure_status': None,
            'histogram': None
        }
    
    hist = cv2.calcHist([sclera_pixels], [0], None, [256], [0, 256])
    mean_brightness = np.mean(sclera_pixels)

    # Classify Exposure Level
    if mean_brightness > 200:
        exposure_status = "Overexposed (Too Bright)"
    elif mean_brightness < 80:
        exposure_status = "Underexposed (Too Dark)"
    else:
        exposure_status = "Well Exposed"

    # Optional: Create visualization plots
    if plot_results:
        # Show the extracted sclera region
        plt.figure(figsize=(6, 6))
        plt.imshow(sclera_region)
        plt.title("Extracted Sclera Region")
        plt.axis('off')
        plt.show()

        # Show histogram of sclera pixel intensities
        plt.figure(figsize=(6, 4))
        plt.plot(hist, color='black')
        plt.title("Histogram of Sclera Intensity (Exposure Analysis)")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()

    return {
        'status': 'success',
        'mean_brightness': float(mean_brightness),
        'exposure_status': exposure_status,
        'histogram': hist.flatten().tolist()
    }