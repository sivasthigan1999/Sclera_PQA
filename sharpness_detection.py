import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def analyze_sclera_quality(eye_masks, iris_masks, image_path, show_visualization=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if eye_masks is not None and iris_masks is not None:
        # Extract eye mask
        eye_mask = (eye_masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        eye_mask = cv2.resize(eye_mask, (image.shape[1], image.shape[0]))

        # Extract iris mask
        iris_mask = (iris_masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        iris_mask = cv2.resize(iris_mask, (image.shape[1], image.shape[0]))

        # Step 3: Subtract iris mask from eye mask to obtain sclera region
        sclera_mask = cv2.subtract(eye_mask, iris_mask)

        # Step 4: Apply morphological closing to refine the sclera mask
        kernel = np.ones((5,5), np.uint8)
        sclera_mask = cv2.morphologyEx(sclera_mask, cv2.MORPH_CLOSE, kernel)

        # Step 5: Extract the sclera region from the original image
        sclera_region = cv2.bitwise_and(image_rgb, image_rgb, mask=sclera_mask)

        # Step 6: Analyze the sclera
        gray_sclera = cv2.cvtColor(sclera_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_sclera, cv2.CV_64F).var()
        red_channel = sclera_region[:, :, 2]
        redness_score = np.mean(red_channel[sclera_mask > 0])

        # Step 7: Optional visualization
        if show_visualization:
            plt.figure(figsize=(6, 6))
            plt.imshow(sclera_region)
            plt.title("Sclera Region (After Iris Removal)")
            plt.axis('off')
            plt.show()

        return {
            'sharpness_score': laplacian_var,
            'redness_score': redness_score,
            'success': True,
            'gray_sclera': gray_sclera,
            'sclera_region': sclera_region,
            'sclera_mask': sclera_mask
        }
    else:
        return {
            'sharpness_score': 0,
            'redness_score': 0,
            'success': False
        }

# Example usage
# image_path = "testimages/Hari_good_sclera.jpg"
# eye_model_path = "models/eye.pt"
# iris_model_path = "models/iris.pt"
# result = analyze_sclera_quality(image_path, eye_model_path, iris_model_path, show_visualization=True)
# print(result)