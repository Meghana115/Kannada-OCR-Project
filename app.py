import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image

# 1. Initialize EasyOCR with Kannada and English support
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['kn', 'en'])

reader = load_ocr()

# Set up the Streamlit UI headers matching your Capstone Theme
st.title("ಕನ್ನಡ ಸಂಖ್ಯಾ ಗುರುತಿಸುವಿಕೆ ವ್ಯವಸ್ಥೆ")
st.subheader("Universal Kannada Text & Numeral Recognition Engine")

uploaded_file = st.file_uploader("Upload an image, story, screenshot, or Wikipedia clip:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image and then to OpenCV format
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Uploaded Image", use_container_width=True)
    
    # Convert PIL to openCV numpy array (RGB to BGR)
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Already grayscale
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # --- ADVANCED PREPROCESSING PIPELINE ---
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_img = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    with st.expander("Show Cleaner Preprocessed Image (What the AI Sees)"):
        st.image(processed_img, caption="Binarized Text Layout", channels="GRAY", use_container_width=True)
        
    # --- RUN THE OCR ENGINE ---
    with st.spinner("Analyzing text blocks, characters, and numerals..."):
        try:
            results = reader.readtext(processed_img)
            
            if len(results) == 0:
                st.warning("No text or numerals detected. Try adjusting the image lighting.")
            else:
                st.success("Extraction Complete!")
                
                extracted_text_lines = []
                detected_numbers = []
                
                kannada_digits = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯']
                numeral_words = ['ಒಂದು', 'ಎರಡು', 'ಮೂರು', 'ನಾಲ್ಕು', 'ಐದು', 'ಆರು', 'ಏಳು', 'ಎಂಟು', 'ಒಂಬತ್ತು', 'ಹತ್ತು']
                
                for (bbox, text, prob) in results:
                    extracted_text_lines.append(text)
                    
                    # Split lines into individual words to look for numbers
                    words = text.split()
                    for word in words:
                        clean_word = word.strip(".,\"'()!:;-")
                        
                        # Match digits or specific handwritten/printed word equivalents
                        has_digit = any(digit in clean_word for digit in kannada_digits)
                        is_numeral_word = clean_word in numeral_words
                        
                        if has_digit or is_numeral_word:
                            if clean_word not in detected_numbers:
                                detected_numbers.append(clean_word)

                # --- DISPLAY RESULTS PANEL ---
                st.markdown("### 📝 Extracted Full Text / Story Content:")
                full_story = "\n".join(extracted_text_lines)
                st.text_area("Recognized Script Output:", value=full_story, height=250)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Detected Character Blocks", value=len(results))
                with col2:
                    st.metric(label="Confidence Score Avg", value=f"{np.mean([res[2] for res in results])*100:.1f}%")
                
                if detected_numbers:
                    st.markdown("### 🔢 Identified Numeral Elements:")
                    for item in detected_numbers:
                        st.info(f"Detected Numeral/Word Token: **{item}**")
                        
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.info("Tip: If the app stalls, reboot your Streamlit server instance via the dashboard.")
