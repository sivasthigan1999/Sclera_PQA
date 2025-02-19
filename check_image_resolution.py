from PIL import Image

def check_image_resolution(image_path, min_width, min_height):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image Resolution: {width}x{height}")
            return width >= min_width and height >= min_height
    except Exception as e:
        print(f"Error: {e}")
        return False

# # Example Usage
# image_path = "/content/Screenshot 2025-01-08 at 21.41.17.png"
# min_width = 2500
# min_height = 1500

# if check_image_resolution(image_path, min_width, min_height):
#     print("Image resolution is sufficient.")
# else:
#     print("Image resolution is too low.")
