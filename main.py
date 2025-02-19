import warnings

# Filter out the specific RuntimeError warning thrown by PyTorch's custom internals.
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")

import streamlit as st
from check_image_resolution import check_image_resolution
from crop_and_find_center_of_iris import find_iris_center
from crop_and_find_center_of_eye import find_eye_center
from check_eye_direction import determine_eye_direction
from sharpness_detection import analyze_sclera_quality
from check_exposure import analyze_sclera_exposure
from circularity_check import analyze_iris_shape
from check_glare import analyze_glare   
import cv2
import pywt  # PyWavelets
from io import BytesIO
from PIL import Image, ExifTags


def main():
    st.title("Sclera PQA")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an eye image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read the file bytes and open the image via PIL
        file_bytes = uploaded_file.read()
        image = Image.open(BytesIO(file_bytes))
        
        # Convert to RGB if the image mode is RGBA or LA to prevent saving errors
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        
        # Correct orientation using EXIF data (if available)
        try:
            # Check if image has an EXIF method
            if hasattr(image, '_getexif'):
                exif = image._getexif()
            elif hasattr(image, 'getexif'):
                exif = image.getexif()
            else:
                exif = None

            if exif:
                for tag in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[tag] == 'Orientation':
                        orientation_tag = tag
                        break

                exif = dict(exif.items()) if not isinstance(exif, dict) else exif
                orientation_value = exif.get(orientation_tag, None)

                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
        except Exception as e:
            st.write("Could not determine EXIF orientation:", e)
        
        # Create columns for image display and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save the corrected image temporarily for further processing
        image_path = "temp_uploaded_image.jpg"
        image.save(image_path)
        
        # Model paths
        iris_model_path = "models/iris.pt"
        eye_model_path = "models/eye.pt"
        
        with col2:
            st.subheader("Analysis Results")
            
            # Adjustable threshold sliders for the image resolution
            min_width = st.slider("Set Minimum Image Width", min_value=500, max_value=5000, value=2500, step=100)
            min_height = st.slider("Set Minimum Image Height", min_value=500, max_value=5000, value=1500, step=100)
            
            # Resolution check
            resolution_check = check_image_resolution(image_path, min_width, min_height)
            width, height = image.size
            st.write("Image Resolution:", f"{width} x {height}")
            
            if not resolution_check:
                st.error(f"The resolution of the image should be above the threshold value: {min_width} x {min_height}")
            else:
                st.write("Resolution Check: ✅ Sufficient")
            
            # Iris Analysis
            with st.spinner('Analyzing iris...'):
                iris_results = find_iris_center(image_path, iris_model_path)
                if len(iris_results) < 3:
                    st.error("No iris mask detected retake the photo")
                    return  # Stop further processing if no iris mask is detected
                iris_center_x, iris_center_y, iris_masks = iris_results
                
                image_cv = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                # Add adjustable threshold sliders for iris shape analysis
                circularity_threshold = st.slider(
                    "Set Circularity Threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.01
                )
                aspect_ratio_lower = st.slider(
                    "Set Aspect Ratio Lower Bound", min_value=0.0, max_value=2.0, value=0.9, step=0.01
                )
                aspect_ratio_upper = st.slider(
                    "Set Aspect Ratio Upper Bound", min_value=0.0, max_value=2.0, value=1.8, step=0.01
                )
                aspect_ratio_range = (aspect_ratio_lower, aspect_ratio_upper)
                
                # Use the adjustable thresholds when calling analyze_iris_shape
                iris_shape_results = analyze_iris_shape(
                    image_rgb, iris_masks,
                    circularity_threshold=circularity_threshold,
                    aspect_ratio_range=aspect_ratio_range
                )
                
                st.write("### Iris Analysis")
                if iris_shape_results["status"] == "success":
                    metrics = {
                        "Circularity": f"{iris_shape_results['circularity']:.4f}",
                        "Aspect Ratio": f"{iris_shape_results['aspect_ratio']:.4f}",
                        "Edge Detection": 'Complete' if iris_shape_results['has_complete_edge'] else 'Incomplete',
                        "Validation": '✅ Valid' if iris_shape_results['is_valid'] else '❌ Invalid'
                    }
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value}")
                    
                    # Additional check for iris circularity
                    if not iris_shape_results["is_valid"]:
                        st.error("Iris is not circular")
                else:
                    st.error(f"Iris shape analysis failed: {iris_shape_results['message']}")
            
            # Eye Direction Analysis
            with st.spinner('Analyzing eye direction...'):
                eye_center_x, eye_center_y, eye_masks = find_eye_center(image_path, eye_model_path)
                
                # Add adjustable slider for eye direction threshold
                eye_direction_threshold = st.slider(
                    "Set Eye Direction Threshold (px)", min_value=0, max_value=200, value=80, step=10
                )
                
                # Use the adjustable threshold in the eye direction calculation
                eye_direction = determine_eye_direction(
                    iris_center_x, iris_center_y, eye_center_x, eye_center_y, threshold=eye_direction_threshold
                )
                
                st.write("### Eye Direction")
                st.write(f"Position: {eye_direction}")
                
                # Display additional instructions based on eye direction
                if eye_direction.lower() == "centered":
                    st.write("✅ Correct eye direction")
                else:
                    st.write("❌ Incorrect eye direction")
                if eye_direction.lower() == "left":
                    st.info("Move the iris a littlebit Right.")
                elif eye_direction.lower() == "right":
                    st.info("Move the iris a littlebit Left.")
                elif eye_direction.lower() == "up":
                    st.info("Move the iris a littlebit Down.")
                elif eye_direction.lower() == "Down":
                    st.info("Move the iris a littlebit Up.")

            
            # Sclera Analysis
            with st.spinner('Analyzing sclera quality...'):
                sclera_analysis = analyze_sclera_quality(eye_masks, iris_masks, image_path)
                
                st.write("### Sclera Analysis")
                if sclera_analysis['success']:
                    st.write("#### Sclera Sharpness")
                    st.write(f"Sharpness Score: {sclera_analysis['sharpness_score']:.2f}")
                    # st.write(f"Redness Score: {sclera_analysis['redness_score']:.2f}")

                                        # Set thresholds for sharpness and redness
                    sharpness_threshold = st.slider("Set Sharpness Threshold", min_value=0.0, max_value=100.0, value=20.0, step=0.5
)
                    redness_threshold = 200      # Example threshold; lower redness is better
                    
                    if (sclera_analysis['sharpness_score'] >= sharpness_threshold and 
                        sclera_analysis['redness_score'] <= redness_threshold):
                        st.info("Selera sharpness is Good")
                        st.write("Sclera Sharpness: ✅ Good")

                    else:
                        st.warning("Sclera is blur")
                        st.write("Sclera Sharpness: ❌ Bad")
                    # Sclera Exposure Analysis Section
                    gray_sclera = sclera_analysis['gray_sclera']
                    sclera_mask = sclera_analysis['sclera_mask']
                    sclera_region = sclera_analysis['sclera_region']
                    results = analyze_sclera_exposure(gray_sclera, sclera_region, sclera_mask, plot_results=False)
                    st.write("#### Sclera Exposure Analysis")
                    if results['status'] == 'success':
                        st.write(f"Mean Brightness: {results['mean_brightness']:.2f}")
                        if results['exposure_status'] == "Well Exposed":
                            st.write("Sclera Exposure: ✅ Well Exposed")
                        elif results['exposure_status'] == "Overexposed (Too Bright)":
                            st.write("Sclera Exposure: ❌ Overexposed")
                            st.info("please reduce the exposure of the image")

                        elif results['exposure_status'] == "Underexposed (Too Dark)":
                            st.write("Sclera Exposure: ❌ Underexposed")
                            st.info("please increase the exposure of the image")
                    else:
                        st.error(results['message'])
                    
                    # Glare Analysis Section
                    glare_threshold = st.slider("Set Glare Percentage Threshold", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
                    glare_analysis = analyze_glare(gray_sclera)
                    st.write("#### Glare Analysis")
                    st.write(f"Glare Percentage: {glare_analysis:.2f}%")
                    if glare_analysis > glare_threshold:
                        st.write("Glare: ❌ High")
                        st.info("please reduce the glare of the image")
                    else:
                        st.write("Glare: ✅ Low")
                        st.info("Glare is acceptable")

                else:
                    st.error("Eye or iris mask not detected properly!")

if __name__ == "__main__":
    main()
