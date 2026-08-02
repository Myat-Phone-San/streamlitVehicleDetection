import os
import cv2
import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr

# Page Configuration
st.set_page_config(
    page_title="AUTOMATIC VEHICLE IDENTIFICATION SYSTEM (ANPR)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling (Fixed text colors for clear visibility in dark mode)
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000 !important;
    }
    p, span, label, .stRadio label, .stCheckbox label {
        color: #000000 !important;
    }
    .ocr-plate-display {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: 800;
        color: #58a6ff;
        background: #ffffff;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #30363d;
        text-align: center;
        letter-spacing: 2px;
    }
    .active-plate {
        color: #39d353 !important;
        border-color: #238636 !important;
       
    }
    .status-badge-valid {
        background: #ffffff;
        color: #000000;
        border: 1px solid #238636;
        padding: 10px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .status-badge-invalid {
        background: #ffffff;
        color: #000000;
        border: 1px solid #da3633;
        padding: 10px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODELS & DATABASE -----------------
@st.cache_resource
def load_models():
    model_plate = YOLO('best.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return model_plate, ocr_reader

with st.spinner("Loading AI Models (YOLO & EasyOCR)... Please wait."):
    model_plate, ocr_reader = load_models()

DB_PATH = 'Car_List_MinDaMa_overall.csv'

def get_db():
    if os.path.exists(DB_PATH):
        try:
            if DB_PATH.endswith('.csv'):
                return pd.read_csv(DB_PATH)

            xls = pd.ExcelFile(DB_PATH)
            all_dfs = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(DB_PATH, sheet_name=sheet)
                if 'Car No.' not in df.columns and 'Car Number' not in df.columns and 'Room No.' not in df.columns:
                    df = pd.read_excel(DB_PATH, sheet_name=sheet, header=1)
                all_dfs.append(df)
            combined_df = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]
            return combined_df
        except Exception as e:
            st.error(f"Error reading database: {e}")
    return None

def get_best_plate_crop(image):
    results = model_plate(image, verbose=False, conf=0.45)[0]
    if len(results.boxes) == 0:
        return None

    best_box = None
    max_conf = -1.0

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w_box, h_box = x2 - x1, y2 - y1
        aspect_ratio = w_box / float(h_box) if h_box > 0 else 0

        if 1.8 <= aspect_ratio <= 4.2:
            if conf > max_conf:
                max_conf = conf
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    h_img, w_img = image.shape[:2]
    padding = 5
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w_img, x2 + padding)
    y2_pad = min(h_img, y2 + padding)

    return image[y1_pad:y2_pad, x1_pad:x2_pad]

def preprocess_and_read_plate(crop_img):
    if crop_img is None or crop_img.size == 0: 
        return "Not detected"

    h, w = crop_img.shape[:2]
    top_part = crop_img[0:int(h * 0.5), 0:w]
    bottom_part = crop_img[int(h * 0.3):h, 0:w]

    def process_sub_img(img):
        resized = cv2.resize(img, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(filtered)

    top_processed = process_sub_img(top_part)
    bot_processed = process_sub_img(bottom_part)

    res_top = ocr_reader.readtext(top_processed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
    res_bot = ocr_reader.readtext(bot_processed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')

    top_text = " ".join(res_top).upper()
    bot_text = " ".join(res_bot).upper()

    noise_words = ["HONDA", "INSIGHT", "TOYOTA", "SUZUKI", "CARRY", "PROBOX", "NISSAN", "FIT", "JUSGM", "QOLIL", "AUDI", "CHANGAN", "DEEPAL", "COROLLA", "FIELDER"]
    for word in noise_words:
        top_text = top_text.replace(word, "")
        bot_text = bot_text.replace(word, "")

    combined_text = f"{top_text} {bot_text}"

    city_code = "YGN"
    if re.search(r'\b(NPW|NPT)\b', combined_text): 
        city_code = "NPW"
    elif re.search(r'\b(MDY|MOY|MDV|MAY)\b', combined_text): 
        city_code = "MDY"
    elif re.search(r'\b(AYY|AVY|AAY)\b', combined_text): 
        city_code = "AYY"
    elif re.search(r'\b(BGO|3GO|8GO)\b', combined_text): 
        city_code = "BGO"
    elif re.search(r'\b(SHN|SHAN)\b', combined_text): 
        city_code = "SHN"
    elif re.search(r'\b(YGN|YON|YCN|VGN)\b', combined_text): 
        city_code = "YGN"

    prefix = ""
    digits = ""

    match_plate = re.search(r'([A-Z0-9]{1,3})[-]?(\d{4})', bot_text)
    if match_plate:
        prefix = match_plate.group(1)
        digits = match_plate.group(2)
    else:
        match_plate_comb = re.search(r'([A-Z0-9]{1,3})[-]?(\d{4})', combined_text)
        if match_plate_comb:
            prefix = match_plate_comb.group(1)
            digits = match_plate_comb.group(2)
        else:
            digits_match = re.findall(r'\b\d{4}\b', bot_text)
            if digits_match:
                digits = digits_match[-1]
            pfx_match = re.search(r'\b([A-Z0-9]{2,3})\b', bot_text)
            if pfx_match:
                prefix = pfx_match.group(1)

    def correct_prefix(pfx):
        pfx = pfx.upper().strip()
        pfx = re.sub(r'[^A-Z0-9]', '', pfx)
        if len(pfx) >= 2:
            chars = list(pfx)
            # First character corrections
            if chars[0] in ['I', 'L', '|', '!']: chars[0] = '1'
            elif chars[0] in ['b']: chars[0] = '6'
            elif chars[0] == 'O': chars[0] = '0'
            elif chars[0] == 'Z': chars[0] = '2'
            elif chars[0] == 'S': chars[0] = '5'
            elif chars[0] == '6': chars[0] = 'G'
                
            # Second character corrections (mapping 6, b, o to G for prefixes like GG)
            if chars[1] in ['6', 'b', 'o', '9']: chars[1] = 'G'
            elif chars[1] == '0': chars[1] = 'D'
            elif chars[1] in ['1', '|']: chars[1] = 'I'
            elif chars[1] == '2': chars[1] = 'Z'
            elif chars[1] == '5': chars[1] = 'S'
            return "".join(chars)
        return pfx

    if prefix:
        prefix = correct_prefix(prefix)

    if prefix and digits:
        return f"{city_code} {prefix}-{digits}"
    elif digits:
        return f"{city_code} {digits}"

    return "Not detected"

def draw_single_best_box(img):
    results = model_plate(img, verbose=False, conf=0.45)[0]
    if len(results.boxes) == 0:
        return img

    best_box = None
    max_conf = -1.0

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / float(bh) if bh > 0 else 0

        if 1.8 <= aspect <= 4.2:
            if conf > max_conf:
                max_conf = conf
                best_box = (x1, y1, x2, y2)

    if best_box is None and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_box = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img

# ----------------- UI LAYOUT (HEADER) -----------------
col_title, col_reset = st.columns([8, 2])
with col_title:
    st.markdown("### 🚗 AUTOMATIC VEHICLE IDENTIFICATION SYSTEM (ANPR)")
with col_reset:
    if st.button("🧹 RESET / CLEAR", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown("---")

# Main Layout: 2 Columns (Left: Viewport / Right: Control Panel)
main_col, side_col = st.columns([1.3, 1])

with main_col:
    st.markdown("##### 📷 Live Viewport / Media Display")
    viewport_placeholder = st.empty()

with side_col:
    st.markdown("##### ⚙️ Control Panel")
    
    input_mode = st.radio("Select Input Source:", ["Upload File", "Live Webcam"], horizontal=True)
    
    img_array = None
    
    if input_mode == "Upload File":
        uploaded_file = st.file_uploader("Upload Image/Video", type=['png', 'jpg', 'jpeg', 'webp'])
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        camera_file = st.camera_input("Take a photo with webcam")
        if camera_file is not None:
            file_bytes = np.frombuffer(camera_file.getvalue(), np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    show_box = st.checkbox("🟢 Show Bounding Box", value=False)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        run_crop = st.button("✂️ CROP PLATE", use_container_width=True)
    with col_b2:
        run_ocr = st.button("🔍 RUN OCR", use_container_width=True)
    with col_b3:
        run_verify = st.button("🛡️ VERIFY", use_container_width=True)

    if img_array is not None:
        display_img = img_array.copy()
        if show_box:
            display_img = draw_single_best_box(display_img)
        viewport_placeholder.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        viewport_placeholder.info("System Standby. Open Camera or Upload File...")

    # --- RESULTS SECTION ---
    st.markdown("---")
    
    st.markdown("**PLATE ROI CROP**")
    crop_placeholder = st.empty()
    
    st.markdown("**DETECTED LICENSE PLATE**")
    ocr_log_placeholder = st.empty()
    
    if 'plate_result' not in st.session_state:
        st.session_state.plate_result = "READY..."
    if 'cropped_roi' not in st.session_state:
        st.session_state.cropped_roi = None

    if img_array is not None and run_crop:
        with st.spinner("Cropping license plate..."):
            roi = get_best_plate_crop(img_array)
            if roi is not None and roi.size > 0:
                st.session_state.cropped_roi = roi
                roi_resized = cv2.resize(roi, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                crop_placeholder.image(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                crop_placeholder.warning("No plate region detected for cropping.")
                st.session_state.cropped_roi = None

    if img_array is not None and run_ocr:
        with st.spinner("Processing OCR..."):
            roi = st.session_state.cropped_roi if st.session_state.cropped_roi is not None else get_best_plate_crop(img_array)
            if roi is not None and roi.size > 0:
                final_plate_str = preprocess_and_read_plate(roi)
                st.session_state.plate_result = final_plate_str
            else:
                st.session_state.plate_result = "Not detected"

    is_active = st.session_state.plate_result not in ["READY...", "Not detected", "No media frame loaded"]
    active_class = "active-plate" if is_active else ""
    ocr_log_placeholder.markdown(f'<div class="ocr-plate-display {active_class}">{st.session_state.plate_result}</div>', unsafe_allow_html=True)

    # Database Verification Panel
    st.markdown("**DATABASE VERIFICATION DETAILS**")
    db_result_placeholder = st.empty()

    if run_verify:
        detected_plate = st.session_state.plate_result
        if not detected_plate or detected_plate in ["READY...", "Not detected"]:
            st.warning("Please run OCR recognition first before verifying status.")
        else:
            det_digits = "".join(re.findall(r'\d+', detected_plate))
            det_suffix = det_digits[-4:] if len(det_digits) >= 4 else None

            if not det_suffix:
                db_result_placeholder.error("Clear character matching error (4 digits required)")
            else:
                df = get_db()
                if df is None:
                    db_result_placeholder.error("Database lookup failed")
                else:
                    car_col = None
                    for col in df.columns:
                        if 'car' in str(col).lower():
                            car_col = col
                            break
                    if not car_col:
                        car_col = df.columns[-1]

                    match_row = None
                    for _, row in df.iterrows():
                        db_car_no = str(row[car_col])
                        db_digits = "".join(r for r in db_car_no if r.isdigit())
                        if db_digits and db_digits.endswith(det_suffix):
                            match_row = row.to_dict()
                            break

                    if match_row:
                        clean_data = {}
                        target_keys = []
                        other_keys = []
                        
                        for k in match_row.keys():
                            k_lower = str(k).strip().lower()
                            if any(x in k_lower for x in ['sn', 'sn.', 's/n', 'sr. no.', 'sr no']):
                                continue
                            if 'car' in k_lower or 'room' in k_lower:
                                target_keys.append(k)
                            else:
                                other_keys.append(k)
                                
                        sorted_keys = target_keys + other_keys

                        for k in sorted_keys:
                            v = match_row[k]
                            if pd.notna(v) and str(v).strip() != "":
                                if isinstance(v, float) and v.is_integer():
                                    clean_data[str(k)] = str(int(v))
                                else:
                                    clean_data[str(k)] = str(v)
                        
                        html_out = '<div class="status-badge-valid">✓ REGISTERED VEHICLE</div>'
                        html_out += '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px;">'
                        for k, v in clean_data.items():
                            html_out += f'''
                                <div style="background: #0d1117; padding: 6px; border-radius: 4px; border: 1px solid #30363d;">
                                    <div style="font-size: 0.65rem; color: #8b949e; text-transform: uppercase;">{k}</div>
                                    <div style="font-size: 0.85rem; font-weight: 600; color: #58a6ff;">{v}</div>
                                </div>'''
                        html_out += '</div>'
                        db_result_placeholder.markdown(html_out, unsafe_allow_html=True)
                    else:
                        db_result_placeholder.markdown(f'''
                            <div class="status-badge-invalid">✕ NOT FOUND / UNREGISTERED</div>
                            <div style="color: #8b949e; font-size: 0.8rem; text-align: center;">No matching record found for digits: {det_suffix}</div>
                        ''', unsafe_allow_html=True)
    else:
        db_result_placeholder.info("Run Database Check to display owner details...")