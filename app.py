import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

st.set_page_config(page_title="SafetyNet Enterprise", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .meta-box { background-color: #1c1f26; padding: 15px; border-radius: 8px; border-left: 5px solid #00ff99; margin-bottom: 10px; font-family: monospace; color: white;}
    .critical-box { background-color: #ff3333; color: white; padding: 20px; border-radius: 8px; font-weight: bold; font-size: 22px; text-align: center; box-shadow: 0 0 20px #ff0000;}
    .warning-box { background-color: #ffaa00; color: black; padding: 20px; border-radius: 8px; font-weight: bold; font-size: 22px; text-align: center; box-shadow: 0 0 15px #ffaa00;}
    .safe-box { background-color: #00cc66; color: white; padding: 20px; border-radius: 8px; font-weight: bold; font-size: 22px; text-align: center; box-shadow: 0 0 15px #00cc66;}
    .stat-card { background: linear-gradient(135deg, #1c1f26, #2a2d36); padding: 20px; border-radius: 10px; text-align: center; color: white; border: 1px solid #00ff99;}
    h1 {color: #00ff99 !important;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 🔥 NEW MODEL - 7 CLASSES (97.8% mAP@50)
# Classes: Gloves, Helmet, Non-Helmet, Person, Shoes, Vest, bare-arms

IMAGE_THRESHOLDS = {
    "Helmet":     0.40,
    "Vest":       0.45,
    "Person":     0.45,
    "Shoes":      0.45,
    "Gloves":     0.40,
    "Non-Helmet": 0.40,
    "bare-arms":  0.40
}

LIVE_THRESHOLDS = {
    "Helmet":     0.35,
    "Vest":       0.45,
    "Person":     0.40,
    "Shoes":      0.40,
    "Gloves":     0.35,
    "Non-Helmet": 0.35,
    "bare-arms":  0.35
}

MASTER_CONF_IMAGE = 0.30
MASTER_CONF_LIVE = 0.30

CRITICAL_GEAR = ["Helmet", "Vest"]
SECONDARY_GEAR = ["Shoes", "Gloves"]

CLASS_COLORS = {
    "Person":     (255, 200, 0),
    "Helmet":     (0, 255, 100),
    "Vest":       (50, 180, 255),
    "Gloves":     (255, 100, 200),
    "Shoes":      (200, 100, 255),
    "Non-Helmet": (0, 50, 255),
    "bare-arms":  (0, 100, 255)
}

def smart_filter(results, model_names, thresholds):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return [], results
    keep_indices = []
    detected = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        cls_name = model_names[cls_id]
        conf = float(boxes.conf[i])
        threshold = thresholds.get(cls_name, 0.5)
        if conf >= threshold:
            keep_indices.append(i)
            detected.append(cls_name)
    if len(keep_indices) > 0:
        results[0].boxes = boxes[keep_indices]
    else:
        results[0].boxes = boxes[:0]
    return detected, results

def check_compliance(detected_classes):
    missing_critical = []
    missing_secondary = []
    
    if "Non-Helmet" in detected_classes:
        missing_critical.append("Helmet")
    elif "Helmet" not in detected_classes:
        missing_critical.append("Helmet")
    
    if "Vest" not in detected_classes:
        missing_critical.append("Vest")
    
    if "Gloves" not in detected_classes:
        missing_secondary.append("Gloves")
    
    if "Shoes" not in detected_classes:
        missing_secondary.append("Shoes")
    
    if "bare-arms" in detected_classes:
        missing_secondary.append("Long Sleeves")
    
    return missing_critical, missing_secondary

def draw_sleek_boxes(frame, results, model_names):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        cls_id = int(boxes.cls[i])
        cls_name = model_names[cls_id]
        conf = float(boxes.conf[i])
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        corner_len = 15
        thickness = 3
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
        label = f"{cls_name.upper()} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

if "violation_log" not in st.session_state:
    st.session_state.violation_log = []

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1973/1973617.png", width=80)
st.sidebar.title("🛡️ COMMAND CENTER")
app_mode = st.sidebar.radio("Operation Mode:", 
    ["📊 Dashboard (Image)", "🎞️ Forensic Scan (Video)", "🔴 Live Surveillance", "📋 Violation Log"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Model Performance")
st.sidebar.success("""
**PRODUCTION GRADE AI**

✅ mAP@50: **97.8%**  
✅ Precision: **97.6%**  
✅ Recall: **94.9%**  
✅ 9,861 training images  
✅ 7 PPE Classes
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Detection Classes")
st.sidebar.markdown("""
**🔴 Critical:** Helmet, Vest  
**🟡 Secondary:** Shoes, Gloves  
**⚠️ Auto-Detect:** Non-Helmet, Bare-Arms  
**👤 Tracking:** Personnel
""")

st.title("🛡️ SafetyNet AI: Industrial Compliance Monitor")
st.caption("Enterprise PPE Detection | YOLOv8 | 97.8% mAP@50 | Production Ready")

if app_mode == "📊 Dashboard (Image)":
    uploaded_file = st.file_uploader("📤 Upload Site Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='meta-box'><b>📐 Resolution</b><br>{img.size[0]} x {img.size[1]}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='meta-box'><b>🎨 Format</b><br>{img.format}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='meta-box'><b>💾 Size</b><br>{uploaded_file.size/1024:.1f} KB</div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='meta-box'><b>🎯 Mode</b><br>Color: {img.mode}</div>", unsafe_allow_html=True)

        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        with st.spinner("🧠 AI Analyzing Frame..."):
            results = model.predict(img_cv2, conf=MASTER_CONF_IMAGE, verbose=False)
            detected_classes, results = smart_filter(results, model.names, IMAGE_THRESHOLDS)
            res_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        st.markdown("### 📊 Detection Summary")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Person')}</h2>👤 Persons</div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Helmet')}</h2>⛑️ Helmets</div>", unsafe_allow_html=True)
        s3.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Vest')}</h2>🦺 Vests</div>", unsafe_allow_html=True)
        s4.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Gloves')}</h2>🧤 Gloves</div>", unsafe_allow_html=True)
        s5.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Shoes')}</h2>👟 Shoes</div>", unsafe_allow_html=True)
        s6.markdown(f"<div class='stat-card'><h2>{detected_classes.count('Non-Helmet') + detected_classes.count('bare-arms')}</h2>⚠️ Violations</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        if "Person" in detected_classes:
            missing_critical, missing_secondary = check_compliance(detected_classes)
            if missing_critical:
                st.markdown(f"<div class='critical-box'>❌ CRITICAL VIOLATION — MISSING: {', '.join(missing_critical).upper()}</div>", unsafe_allow_html=True)
                st.session_state.violation_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Type": "CRITICAL",
                    "Missing": ", ".join(missing_critical),
                    "Source": "Image Upload"
                })
            elif missing_secondary:
                st.markdown(f"<div class='warning-box'>⚠️ MINOR WARNING — MISSING: {', '.join(missing_secondary).upper()}</div>", unsafe_allow_html=True)
                st.session_state.violation_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Type": "WARNING",
                    "Missing": ", ".join(missing_secondary),
                    "Source": "Image Upload"
                })
            else:
                st.markdown(f"<div class='safe-box'>✅ 100% COMPLIANT — ALL SAFETY GEAR DETECTED</div>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No personnel detected in the frame.")

        st.image(res_rgb, use_container_width=True, caption="Annotated Detection Output")

elif app_mode == "🎞️ Forensic Scan (Video)":
    uploaded_video = st.file_uploader("📤 Upload Evidence Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='meta-box'><b>📐 Resolution</b><br>{int(cap.get(3))}x{int(cap.get(4))}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='meta-box'><b>🎬 FPS</b><br>{fps}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='meta-box'><b>🎞️ Frames</b><br>{total_frames}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='meta-box'><b>⏱️ Duration</b><br>{duration:.1f}s</div>", unsafe_allow_html=True)
        progress = st.progress(0)
        status = st.empty()
        stframe = st.empty()
        violation_counter = {"critical": 0, "warning": 0, "safe": 0}
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            frame_resized = cv2.resize(frame, (800, 600))
            results = model.predict(frame_resized, conf=MASTER_CONF_IMAGE, verbose=False)
            detected_classes, results = smart_filter(results, model.names, IMAGE_THRESHOLDS)
            if "Person" in detected_classes:
                missing_c, missing_s = check_compliance(detected_classes)
                if missing_c: violation_counter["critical"] += 1
                elif missing_s: violation_counter["warning"] += 1
                else: violation_counter["safe"] += 1
            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            stframe.image(annotated)
            progress.progress(min(frame_idx/total_frames, 1.0))
            status.text(f"Processing frame {frame_idx}/{total_frames}")
        cap.release()
        st.success("✅ Scan Complete!")
        v1, v2, v3 = st.columns(3)
        v1.metric("🔴 Critical Frames", violation_counter["critical"])
        v2.metric("🟡 Warning Frames", violation_counter["warning"])
        v3.metric("🟢 Safe Frames", violation_counter["safe"])

elif app_mode == "🔴 Live Surveillance":
    st.markdown("### 🔴 Live Surveillance Terminal")
    st.caption("Modern HUD interface with real-time PPE compliance tracking")
    if "cam_active" not in st.session_state: 
        st.session_state.cam_active = False
    col1, col2 = st.columns(2)
    if not st.session_state.cam_active:
        if col1.button("🔥 LAUNCH TERMINAL", use_container_width=True, type="primary"):
            st.session_state.cam_active = True
            st.rerun() 
    else:
        if col1.button("🛑 STOP TERMINAL", use_container_width=True):
            st.session_state.cam_active = False
            st.rerun()
    os.makedirs("violations", exist_ok=True)
    if st.session_state.cam_active:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        win_name = "SafetyNet HUD (Press Q to Exit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        prev_time = 0
        last_snapshot_time = 0
        try:
            while cap.isOpened() and st.session_state.cam_active:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                results = model.predict(frame, conf=MASTER_CONF_LIVE, verbose=False)
                detected_classes, results = smart_filter(results, model.names, LIVE_THRESHOLDS)
                frame = draw_sleek_boxes(frame, results, model.names)
                status_text = "SCANNING"
                status_color = (150, 150, 150)
                missing_critical = []
                missing_secondary = []
                if "Person" in detected_classes:
                    missing_critical, missing_secondary = check_compliance(detected_classes)
                    if missing_critical:
                        status_text = "CRITICAL"
                        status_color = (60, 60, 255)
                    elif missing_secondary:
                        status_text = "WARNING"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "COMPLIANT"
                        status_color = (0, 220, 100)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 40), (15, 17, 23), -1)
                cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                cv2.line(frame, (0, 40), (w, 40), status_color, 2)
                cv2.putText(frame, "SAFETYNET AI", (15, 27), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 153), 1, cv2.LINE_AA)
                cv2.putText(frame, "|  LIVE", (180, 27), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (265, 22), 5, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (275, 27), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                timestamp = datetime.now().strftime("%H:%M:%S")
                ts_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.putText(frame, timestamp, (w//2 - ts_size[0]//2, 27), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                curr_time = time.time()
                fps = int(1 / (curr_time - prev_time + 0.001))
                prev_time = curr_time
                cv2.putText(frame, f"FPS {fps}", (w - 90, 27), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA)
                panel_x = 15
                panel_y = 60
                panel_w = 240
                panel_h = 280
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                             (15, 17, 23), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                             status_color, 1)
                cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 30), 
                             status_color, -1)
                cv2.putText(frame, f"STATUS: {status_text}", (panel_x + 10, panel_y + 21), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                checklist = [
                    ("Personnel", "Person" in detected_classes),
                    ("Helmet",    "Helmet" in detected_classes and "Non-Helmet" not in detected_classes),
                    ("Vest",      "Vest" in detected_classes),
                    ("Gloves",    "Gloves" in detected_classes),
                    ("Shoes",     "Shoes" in detected_classes),
                    ("Sleeves",   "bare-arms" not in detected_classes if "Person" in detected_classes else False)
                ]
                y_offset = panel_y + 55
                for item_name, detected in checklist:
                    if "Person" not in detected_classes and item_name != "Personnel":
                        icon = "--"
                        color = (120, 120, 120)
                    elif detected:
                        icon = "OK"
                        color = (0, 220, 100)
                    else:
                        icon = "NO"
                        color = (60, 60, 255)
                    cv2.putText(frame, item_name, (panel_x + 15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (panel_x + 170, y_offset - 15), 
                                 (panel_x + 220, y_offset + 5), color, -1)
                    text_size = cv2.getTextSize(icon, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(frame, icon, (panel_x + 170 + (50 - text_size[0])//2, y_offset - 1), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 35
                if "Person" in detected_classes and (missing_critical or missing_secondary):
                    alert_h = 50
                    if missing_critical:
                        alert_text = f"CRITICAL: MISSING {', '.join(missing_critical).upper()}"
                        alert_color = (60, 60, 255)
                        if curr_time - last_snapshot_time > 5:
                            filename = f"violations/violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(filename, frame)
                            last_snapshot_time = curr_time
                    else:
                        alert_text = f"WARNING: MISSING {', '.join(missing_secondary).upper()}"
                        alert_color = (0, 165, 255)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, h - alert_h), (w, h), alert_color, -1)
                    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
                    cv2.putText(frame, alert_text, ((w - text_size[0]) // 2, h - 18), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    st.session_state.cam_active = False
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if not st.session_state.cam_active: 
                st.rerun()

elif app_mode == "📋 Violation Log":
    st.markdown("### 📋 Compliance Violation Records")
    if st.session_state.violation_log:
        df = pd.DataFrame(st.session_state.violation_log)
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Violations", len(df))
        col2.metric("Critical", len(df[df["Type"]=="CRITICAL"]))
        col3.metric("Warnings", len(df[df["Type"]=="WARNING"]))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report (CSV)", csv, "violations_report.csv", "text/csv")
        if st.button("🗑️ Clear Log"):
            st.session_state.violation_log = []
            st.rerun()
    else:
        st.info("No violations recorded yet.")
    if os.path.exists("violations"):
        snaps = sorted(os.listdir("violations"), reverse=True)[:6]
        if snaps:
            st.markdown("### 📸 Recent Violation Snapshots")
            cols = st.columns(3)
            for i, snap in enumerate(snaps):
                cols[i%3].image(f"violations/{snap}", caption=snap)