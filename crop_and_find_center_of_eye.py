import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def find_eye_center(eye_image_path, model_path):
    # Load YOLO model
    my_new_model = YOLO(model_path)

    # Load image
    eye_image = cv2.imread(eye_image_path)
    image_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)

    # YOLO segmentation prediction
    results = my_new_model.predict(eye_image_path, conf=0.5)
    eye_masks = results[0].masks

    if eye_masks is not None:
        eye_mask = (eye_masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        eye_mask = cv2.resize(eye_mask, (image_rgb.shape[1], image_rgb.shape[0]))

        # Visualize YOLO mask overlay
        overlay = image_rgb.copy()
        overlay[eye_mask > 0] = (0, 255, 0)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title("YOLO Mask Overlay")
        plt.axis('off')
        plt.show()

        # Extract polygonal contours
        contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            poly_mask = np.zeros_like(image_rgb, dtype=np.uint8)
            cv2.fillPoly(poly_mask, contours, (255, 255, 255))
            cropped_polygonal = cv2.bitwise_and(image_rgb, poly_mask)
            background_mask = (poly_mask == 0)
            cropped_polygonal[background_mask.all(axis=-1)] = (0, 0, 0)

            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            margin = 30
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image_rgb.shape[1], w + 2 * margin)
            h = min(image_rgb.shape[0], h + 2 * margin)
            cropped_polygonal = cropped_polygonal[y:y + h, x:x + w]

            # Find centroid of the segmented mask
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                eye_center_x = int(M["m10"] / M["m00"]) # Adjusted for cropped image
                eye_center_y = int(M["m01"] / M["m00"]) # Adjusted for cropped image
            else:
                eye_center_x, eye_center_y = w // 2, h // 2  # Default to center if division by zero

            # Draw center point on cropped image
            center_marked_eye_image = cv2.circle(image_rgb, (eye_center_x, eye_center_y), 5, (255, 0, 0), -1)  # Red dot

            # Display the cropped image with center marked
            plt.figure(figsize=(6, 6))
            plt.imshow(center_marked_eye_image)
            plt.title("Polygon-Cropped Image with Center Marked")
            plt.axis('off')
            plt.show()

            # print(f"eye center coordinates: ({eye_center_x}, {eye_center_y}) in cropped image.")
            return eye_center_x, eye_center_y, eye_masks   # Return the coordinates

        else:
            print("No valid contours found!")
            return None, None  # Return None if no contours found
    else:
        print("No mask detected!")
        return None, None  # Return None if no mask detected

# Example usage
# eye_center_x, eye_center_y = find_eye_center("/content/Hari_good_sclera.jpg", "/content/eye.pt")
