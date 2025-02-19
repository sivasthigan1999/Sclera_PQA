from crop_and_find_center_of_iris import find_iris_center  # Import the function
from crop_and_find_center_of_eye import find_eye_center  # Import the function

def determine_eye_direction(iris_center_x, iris_center_y, eye_center_x, eye_center_y,threshold):
    dx = iris_center_x - eye_center_x
    dy = iris_center_y - eye_center_y

    if abs(dx) <= threshold and abs(dy) <= threshold:
        position = "Centered"
    elif dx > threshold:
        position = "Right"
    elif dx < -threshold:
        position = "Left"
    elif dy > threshold:
        position = "Down"
    else:
        position = "Up"

    print(f"Iris Center: ({iris_center_x}, {iris_center_y})")
    print(f"Eye Center: ({eye_center_x}, {eye_center_y})")
    # print(f"Iris Position: {position}")

    return position

# Example usage
# iris_image_path = "/content/Hari_good_sclera.jpg"
# iris_model_path = "/content/iris.pt"

# eye_image_path = "/content/Hari_good_sclera.jpg"
# eye_model_path = "/content/iris.pt"

# # Get iris center coordinates
# iris_center_x, iris_center_y = find_iris_center(iris_image_path, iris_model_path)
# eye_center_x, eye_center_y = find_iris_center(eye_image_path, eye_model_path)


# # Determine eye direction
# position = determine_eye_direction(iris_center_x, iris_center_y, eye_center_x, eye_center_y)