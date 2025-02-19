import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def find_iris_center(image_path, model_path):
    # Load YOLO model for iris segmentation
    my_new_model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # YOLO segmentation prediction
    results = my_new_model.predict(image_path, conf=0.5)
    iris_masks = results[0].masks

    if iris_masks is not None:
        iris_mask = (iris_masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        iris_mask = cv2.resize(iris_mask, (image.shape[1], image.shape[0]))

        # Visualize YOLO mask overlay
        overlay = image_rgb.copy()
        overlay[iris_mask > 0] = (0, 255, 0)  # Green overlay for segmented iris
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title("YOLO Mask Overlay - Iris Segmentation")
        plt.axis('off')
        plt.show()

        # Extract polygonal contours of iris
        contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            poly_mask = np.zeros_like(image_rgb, dtype=np.uint8)
            cv2.fillPoly(poly_mask, contours, (255, 255, 255))  # White mask for the iris
            cropped_iris = cv2.bitwise_and(image_rgb, poly_mask)
            background_mask = (poly_mask == 0)
            cropped_iris[background_mask.all(axis=-1)] = (0, 0, 0)  # Black background

            # Compute bounding box around the segmented iris
            x, y, w, h = cv2.boundingRect(contours[0])
            margin = 10  # Small margin to ensure full iris is included
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            cropped_iris = cropped_iris[y:y + h, x:x + w]

            # Find centroid of the segmented iris mask
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                iris_center_x = int(M["m10"] / M["m00"])   # Adjust for cropped image
                iris_center_y = int(M["m01"] / M["m00"])   # Adjust for cropped image
            else:
                iris_center_x, iris_center_y = w // 2, h // 2  # Default to center if division by zero

            # Draw center point on the cropped iris image
            center_marked_iris_image = cv2.circle(image_rgb, (iris_center_x, iris_center_y), 5, (255, 0, 0), -1)  # Red dot for center

            # Display the cropped iris image with center marked
            plt.figure(figsize=(6, 6))
            plt.imshow(center_marked_iris_image)
            plt.title("Iris Segmentation - Center Marked")
            plt.axis('off')
            plt.show()

            # print(f"Iris center coordinates: ({iris_center_x}, {iris_center_y}) in cropped image.")
            return iris_center_x, iris_center_y, iris_masks
        else:
            print("No valid iris contours found!")
            return None, None
    else:
        print("No iris mask detected!")
        return None, None

# Example usage
# iris_center_x, iris_center_y = find_iris_center("/content/Hari_good_sclera.jpg", "/content/iris.pt")
