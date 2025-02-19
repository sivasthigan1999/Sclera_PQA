import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Load YOLO model for iris segmentation
# model_path = '/content/iris.pt'
# my_new_model = YOLO(model_path)

# # Load image
# new_image = "/content/Hari_good_sclera.jpg"
# image = cv2.imread(new_image)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # YOLO segmentation prediction
# results = my_new_model.predict(new_image, conf=0.5)
# masks = results[0].masks

def compute_iris_circularity(mask):
    """
    Computes the circularity and aspect ratio of the iris.
    
    :param mask: Binary mask of the segmented iris (numpy array).
    :return: (Circularity, Aspect Ratio)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0  # No valid contour

    # Get the largest contour (assumed to be the iris)
    iris_contour = max(contours, key=cv2.contourArea)

    # Compute area and perimeter
    area = cv2.contourArea(iris_contour)
    perimeter = cv2.arcLength(iris_contour, True)

    # Compute circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Compute aspect ratio
    x, y, w, h = cv2.boundingRect(iris_contour)
    aspect_ratio = float(w) / h  # Ideal value for a circle is close to 1

    return circularity, aspect_ratio

def check_hough_circles(image):
    """
    Detects if the iris has a complete circular edge using Hough Circle Transform.

    :param image: Grayscale image of the iris.
    :return: True if a circle is detected, False otherwise.
    """
    edges = cv2.Canny(image, 50, 150)  # Edge detection
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    return circles is not None  # True if circles found

def analyze_iris_shape(image_rgb, iris_masks,circularity_threshold, aspect_ratio_range):
    """
    Analyzes the shape characteristics of the iris.
    
    :param image_rgb: RGB image array
    :param iris_masks: Segmentation masks for iris
    :return: Dictionary containing analysis results
    """
    if iris_masks is None:
        return {"status": "error", "message": "No iris mask detected!"}

    mask = (iris_masks.data[0].cpu().numpy() * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

    # Extract contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "No valid iris contours found!"}

    # Get largest contour
    iris_contour = max(contours, key=cv2.contourArea)

    # Compute metrics
    circularity, aspect_ratio = compute_iris_circularity(mask)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    has_complete_edge = check_hough_circles(gray)

    # Validation criteria
    is_circular = circularity > circularity_threshold
    has_good_aspect_ratio = aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]
    iris_valid = is_circular and has_good_aspect_ratio and has_complete_edge

    return {
        "status": "success",
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "has_complete_edge": has_complete_edge,
        "is_valid": iris_valid
    }

# if masks is not None:
#     mask = (masks.data[0].cpu().numpy() * 255).astype(np.uint8)
#     mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

#     # Extract polygonal contours of iris
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         # Get the largest contour (assumed to be the iris)
#         iris_contour = max(contours, key=cv2.contourArea)

        # Compute circularity & aspect ratio
        # circularity, aspect_ratio = compute_iris_circularity(mask)

#         # Convert to grayscale for edge detection
#         gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Detect Hough circles (complete circular edge)
        # has_complete_edge = check_hough_circles(gray)

#         # Compute centroid of the iris
#         M = cv2.moments(iris_contour)
#         if M["m00"] != 0:
#             iris_center_x = int(M["m10"] / M["m00"])
#             iris_center_y = int(M["m01"] / M["m00"])
#         else:
#             iris_center_x, iris_center_y = 0, 0  # Default to 0 if undefined

#         # Compute bounding box of the iris
#         x, y, w, h = cv2.boundingRect(iris_contour)
#         eye_center_x = x + w // 2
#         eye_center_y = y + h // 2

#         # Compare iris center with eye center
#         dx = iris_center_x - eye_center_x
#         dy = iris_center_y - eye_center_y
#         threshold = 5  # Tolerance for center alignment

#         if abs(dx) <= threshold and abs(dy) <= threshold:
#             position = "Centered"
#         elif dx > threshold:
#             position = "Right"
#         elif dx < -threshold:
#             position = "Left"
#         elif dy > threshold:
#             position = "Down"
#         else:
#             position = "Up"

        # **Validation Criteria**
#         circularity_threshold = 0.8
#         aspect_ratio_range = (0.9, 1.1)

#         is_circular = circularity > circularity_threshold
#         has_good_aspect_ratio = aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]

#         iris_valid = is_circular and has_good_aspect_ratio and has_complete_edge

#         # **Display Results**
#         print(f"Iris Center: ({iris_center_x}, {iris_center_y})")
#         print(f"Eye Center: ({eye_center_x}, {eye_center_y})")
#         print(f"Iris Position: {position}")
#         print(f"Iris Circularity: {circularity:.4f}")
#         print(f"Iris Aspect Ratio: {aspect_ratio:.4f}")
#         print(f"Iris Edge Detection: {'Complete' if has_complete_edge else 'Incomplete'}")
#         print(f"Iris Validation: {'Valid' if iris_valid else 'Invalid'}")

#         # **Draw visualizations**
#         output_image = image_rgb.copy()
#         cv2.drawContours(output_image, [iris_contour], -1, (0, 255, 0), 2)  # Green contour
#         cv2.circle(output_image, (iris_center_x, iris_center_y), 6, (0, 0, 255), -1)  # Red dot for iris center
#         cv2.circle(output_image, (eye_center_x, eye_center_y), 6, (255, 0, 0), -1)  # Blue dot for eye center
#         cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow bounding box around iris

#         # **Show the result**
#         plt.figure(figsize=(8, 8))
#         plt.imshow(output_image)
#         plt.title(f"Iris: {position} | Circularity: {circularity:.4f} | {'Valid' if iris_valid else 'Invalid'}")
#         plt.axis('off')
#         plt.show()

#     else:
#         print("No valid iris contours found!")
# else:
#     print("No iris mask detected!")
