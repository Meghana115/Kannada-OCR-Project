import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from gtts import gTTS
from PIL import Image

st.set_page_config(page_title="Kannada Numeral Recognition Engine", layout="centered")

# --- CUSTOM UI THEMING ---
st.markdown("""
    <style>
    .main-title { color: #8B4513; font-weight: bold; font-size: 34px; font-family: 'Segoe UI', sans-serif; }
    .sub-title { color: #2C3E50; font-weight: 600; font-size: 22px; margin-top: 15px; }
    
    div.stButton > button:first-child {
        background-color: #FF9933 !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 2.5rem !important;
        font-size: 16px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #E68218 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">ಕನ್ನಡ ಸಂಖ್ಯಾ ಗುರುತಿಸುವಿಕೆ ವ್ಯವಸ್ಥೆ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title" style="margin-top:-20px; font-size:20px; color:#D35400;">Dedicated Kannada Numeral Recognition & Accessibility Engine</p>', unsafe_allow_html=True)

# --- KANNADA KNOWLEDGE PIPELINE DICTIONARIES ---
AUDIO_PRONUNCIATION_MAP = {
    0: "ಸೊನ್ನೆ", 1: "ಒಂದು", 2: "ಎರಡು", 3: "ಮೂರು", 4: "ನಾಲ್ಕು",
    5: "ಐದು", 6: "ಆರು", 7: "ಏಳು", 8: "ಎಂಟು", 9: "ಒಂಬತ್ತು", 10: "ಹತ್ತು"
}

NATIVE_DIGITS = {
    "೦": 0, "೧": 1, "೨": 2, "೩": 3, "೪": 4, 
    "೫": 5, "೬": 6, "೭": 7, "೮": 8, "೯": 9
}

NUM_WORDS = {
    "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5,
    "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8, "ಒಂಬತ್ತು": 9, "ಹತ್ತು": 10
}

st.markdown('<p class="sub-title">Document Image Processing (ಕನ್ನಡ Script Core)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image containing Kannada digits or number words:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, use_container_width=True, caption="Uploaded Document Source")
    
    if st.button("Analyze Kannada Document", type="primary"):
        with st.spinner("Executing Image Preprocessing & OCR Pipeline..."):
            
            # --- STAGE 1: BILATERAL FILTER & ADAPTIVE THRESHOLDING ---
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            filtered = cv2.bilateralFilter(resized, 9, 75, 75)
            cleaned = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # --- STAGE 2: INITIALIZE EASYOCR ENGINE ---
            reader = easyocr.Reader(['kn', 'en'], gpu=False)
            ocr_results = reader.readtext(cleaned)
            
            detected_items = []
            extracted_text_lines = []
            
            for (bbox, text, prob) in ocr_results:
                clean_line = text.strip()
                extracted_text_lines.append(clean_line)
                
                # Split line segments into clean individual tokens
                words = clean_line.split()
                for word in words:
                    # Strip away layout punctuation symbols
                    clean_word = word.strip(".,\"'()!:;-‘’“”")
                    
                    # Target A: Explicit Native Kannada Symbols
                    for char in clean_word:
                        if char in NATIVE_DIGITS:
                            val = NATIVE_DIGITS[char]
                            detected_items.append((val, f"{char} = {val} (ಕನ್ನಡ Symbol)"))
                        
                        # Target B: Standard Digits
                        elif char.isdigit():
                            val = int(char)
                            detected_items.append((val, f"{char} = {val} (Standard Digit)"))
                    
                    # Target C: Exact Kannada Textual Number Words
                    if clean_word in NUM_WORDS:
                        val = NUM_WORDS[clean_word]
                        detected_items.append((val, f"{clean_word} = {val} (ಕನ್ನಡ Word)"))

            # --- STAGE 3: OUTPUT RENDERER ---
            unique_detections = list(set(detected_items))
            unique_detections.sort(key=lambda x: x[0])
            
            st.markdown("### 📝 Extracted Text Preview:")
            st.text_area("AI Corpus Readout:", value="\n".join(extracted_text_lines), height=150)
            
            st.markdown(f"### Total Unique Detections Found: {len(unique_detections)}")
            
            with st.container():
                st.markdown('<div style="background-color: #fff9f0; padding: 20px; border-left: 5px solid #FF9933; border-radius: 4px;">', unsafe_allow_html=True)
                
                if unique_detections:
                    audio_sequence = []
                    for val, display_text in unique_detections:
                        st.markdown(f"• **{display_text}**")
                        audio_sequence.append(AUDIO_PRONUNCIATION_MAP[val])
                    
                    st.markdown("#### Generated Phonetic Kannada Voice Playback:")
                    speech_string = ", ".join(audio_sequence)
                    tts = gTTS(text=speech_string, lang='kn', slow=False)
                    audio_filename = "kannada_pronunciation.mp3"
                    tts.save(audio_filename)
                    st.audio(audio_filename, format="audio/mp3")
                else:
                    st.write("No Kannada digits or textual tracking words were detected in this document instance.")
                
                st.markdown('</div>', unsafe_allow_html=True)
