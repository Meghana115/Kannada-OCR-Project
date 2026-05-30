import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from gtts import gTTS
from difflib import get_close_matches
import os

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
            
            # --- STAGE 1: CONTRAST NORMALIZATION & IMAGE ENHANCEMENT ---
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Upscale image to sharpen fine strokes of handwritten Kannada letters
            resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            mean_val = np.mean(resized)
            if mean_val < 127:
                cleaned = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            else:
                cleaned = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # --- STAGE 2: EASYOCR KANNADA FRAMEWORK INITIALIZATION ---
            # Initializing solely with 'kn' (Kannada) and English baseline support
            reader = easyocr.Reader(['kn', 'en'], gpu=False)
            ocr_results = reader.readtext(cleaned)
            
            detected_items = []
            valid_word_keys = list(NUM_WORDS.keys())
            
            # Combine all detected strings into a single text sequence for parsing
            combined_text_stream = " ".join([res[1].strip() for res in ocr_results])
            
            # Layer A: Match Native Kannada Symbols (೦-೯) or Standard Digits
            for char in combined_text_stream:
                if char in NATIVE_DIGITS:
                    val = NATIVE_DIGITS[char]
                    if val != 0:
                        detected_items.append((val, f"{char} = {val} (ಕನ್ನಡ Symbol)"))
                elif char.isdigit():
                    val = int(char)
                    if val != 0:
                        detected_items.append((val, f"{char} = {val} (Standard Digit)"))
            
            # Layer B: Match Textual Kannada Number Words (ಒಂದು, ಎರಡು...)
            for target_word in valid_word_keys:
                val = NUM_WORDS[target_word]
                
                if target_word in combined_text_stream:
                    count = combined_text_stream.count(target_word)
                    for _ in range(count):
                        detected_items.append((val, f"{target_word} = {val} (ಕನ್ನಡ Word)"))
                else:
                    # Fuzzy match fallback for slightly distorted handwriting
                    tokens = re.split(r'\s+', combined_text_stream)
                    for t in tokens:
                        t_clean = re.sub(r'[^\w\s]', '', t).strip()
                        if len(t_clean) >= 2:
                            fuzzy_matches = get_close_matches(t_clean, [target_word], n=1, cutoff=0.50)
                            if fuzzy_matches:
                                detected_items.append((val, f"{target_word} = {val} (Fuzzy Word Match)"))
            
            # --- STAGE 3: METRICS RENDERING & TEXT-TO-SPEECH ---
            unique_detections = list(set(detected_items))
            unique_detections.sort(key=lambda x: x[0])
            
            st.markdown(f"### Total Unique Detections Found: {len(unique_detections)}")
            
            with st.container():
                st.markdown('<div style="background-color: #fff9f0; padding: 20px; border-left: 5px solid #FF9933; border-radius: 4px;">', unsafe_allow_html=True)
                
                if unique_detections:
                    audio_sequence = []
                    for val, display_text in unique_detections:
                        st.markdown(f"• **{display_text}**")
                        audio_sequence.append(AUDIO_PRONUNCIATION_MAP[val])
                else:
                    st.write("No Kannada digits or textual tracking words were detected in this document instance.")
                    
                st.markdown("#### Generated Phonetic Kannada Voice Playback:")
                if unique_detections and audio_sequence:
                    speech_string = ", ".join(audio_sequence)
                    # Convert extracted metrics into native spoken Kannada speech audio
                    tts = gTTS(text=speech_string, lang='kn', slow=False)
                    audio_filename = "kannada_pronunciation.mp3"
                    tts.save(audio_filename)
                    st.audio(audio_filename, format="audio/mp3")
                else:
                    st.write("Audio generation skipped.")
                    
                st.markdown('</div>', unsafe_allow_html=True)