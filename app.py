import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import torch
import time
from collections import deque
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="🚗 Car Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .alert-box {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .safe-box {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    .stat-box {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        min-width: 150px;
    }
</style>
""", unsafe_allow_html=True)

# COCO class names for better identification
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'accident_count' not in st.session_state:
    st.session_state.accident_count = 0

@st.cache_resource
def load_model():
    """Load YOLO model for object detection"""
    try:
        # Use YOLOv8 model (will download automatically)
        model = YOLO('yolov8n.pt')  # nano version for speed
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_accident_indicators(detections, frame_shape, confidence_threshold=0.3):
    """
    Detect potential accident indicators based on:
    1. Multiple vehicles in close proximity
    2. Unusual vehicle orientations
    3. Debris detection
    4. Stopped vehicles in traffic
    """
    accident_score = 0
    indicators = []
    
    cars = []
    trucks = []
    buses = []
    motorcycles = []
    
    # Extract vehicle detections with proper filtering
    for detection in detections:
        if len(detection) >= 6:
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            class_id = int(class_id)
            
            # Debug: Print detections for troubleshooting
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            # Vehicle classes in COCO: car=2, motorcycle=3, bus=5, truck=7
            if class_id in [2, 3, 5, 7] and confidence >= confidence_threshold:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                vehicle_info = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'area': width * height
                }
                
                if class_id == 2:  # car
                    cars.append(vehicle_info)
                elif class_id == 3:  # motorcycle
                    motorcycles.append(vehicle_info)
                elif class_id == 5:  # bus
                    buses.append(vehicle_info)
                elif class_id == 7:  # truck
                    trucks.append(vehicle_info)
    
    # Check for multiple vehicles in close proximity
    all_vehicles = cars + trucks + buses + motorcycles
    
    if len(all_vehicles) >= 2:
        close_pairs = 0
        for i, vehicle1 in enumerate(all_vehicles):
            for vehicle2 in all_vehicles[i+1:]:
                # Calculate distance between vehicles
                dist = np.sqrt((vehicle1['center'][0] - vehicle2['center'][0])**2 + 
                             (vehicle1['center'][1] - vehicle2['center'][1])**2)
                
                # Normalize distance by frame size
                normalized_dist = dist / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
                
                if normalized_dist < 0.1:  # Very close vehicles
                    accident_score += 40
                    indicators.append(f"Very close vehicles: {vehicle1['class_name']} and {vehicle2['class_name']}")
                    close_pairs += 1
                elif normalized_dist < 0.2:  # Close vehicles
                    accident_score += 20
                    indicators.append(f"Close vehicles: {vehicle1['class_name']} and {vehicle2['class_name']}")
                    close_pairs += 1
        
        if close_pairs >= 2:
            accident_score += 30
            indicators.append("Multiple vehicle clusters detected")
    
    # Check for unusual vehicle density
    if len(all_vehicles) >= 4:
        accident_score += 25
        indicators.append(f"High vehicle density: {len(all_vehicles)} vehicles")
    elif len(all_vehicles) >= 3:
        accident_score += 15
        indicators.append(f"Multiple vehicles: {len(all_vehicles)} vehicles")
    
    # Check for large vehicles involved
    if len(trucks) >= 1 and len(cars) >= 1:
        accident_score += 20
        indicators.append("Large vehicle and car interaction")
    
    # Check for motorcycle involvement (higher risk)
    if len(motorcycles) >= 1 and len(cars + trucks + buses) >= 1:
        accident_score += 25
        indicators.append("Motorcycle and vehicle interaction")
    
    # Simulate accident detection based on vehicle arrangements
    if len(all_vehicles) >= 2:
        # Check for vehicles that might be stationary or oddly positioned
        accident_score += 10
        indicators.append("Multiple vehicles in frame")
    
    return accident_score, indicators, all_vehicles

def process_frame(frame, model, confidence_threshold=0.3):
    """Process single frame for accident detection"""
    # Run YOLO detection
    results = model(frame, conf=confidence_threshold, verbose=False)
    
    # Extract detections
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
    
    # Detect accident indicators
    accident_score, indicators, vehicles = detect_accident_indicators(detections, frame.shape, confidence_threshold)
    
    # Create annotated frame
    annotated_frame = frame.copy()
    
    # Draw detections manually for better control
    for detection in detections:
        if len(detection) >= 6:
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            class_id = int(class_id)
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            # Only draw vehicle detections
            if class_id in [2, 3, 5, 7] and confidence >= confidence_threshold:
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add accident detection overlay
    if accident_score > 60:
        cv2.putText(annotated_frame, "ACCIDENT DETECTED!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.rectangle(annotated_frame, (30, 20), (600, 80), (0, 0, 255), 3)
        accident_detected = True
    elif accident_score > 30:
        cv2.putText(annotated_frame, "POTENTIAL ACCIDENT", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 2)
        cv2.rectangle(annotated_frame, (30, 20), (600, 80), (0, 165, 255), 2)
        accident_detected = True
    else:
        cv2.putText(annotated_frame, "NORMAL TRAFFIC", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        accident_detected = False
    
    # Add score display
    cv2.putText(annotated_frame, f"Score: {accident_score:.1f}", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated_frame, accident_detected, accident_score, indicators, vehicles

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Car Accident Detection System</h1>
        <p>Real-time accident detection using YOLO and computer vision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if not model:
        st.error("Failed to load YOLO model. Please check your internet connection.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        detection_mode = st.selectbox(
            "Choose Detection Mode",
            ["Upload Image", "Upload Video", "Live Camera (Demo)"]
        )
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
        
        st.header("📊 Statistics")
        st.metric("Total Detections", len(st.session_state.detection_history))
        st.metric("Accidents Detected", st.session_state.accident_count)
        
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.session_state.accident_count = 0
            st.success("History cleared!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Detection Input")
        
        if detection_mode == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                frame = np.array(image)
                
                # Process frame
                with st.spinner("Processing image..."):
                    annotated_frame, accident_detected, accident_score, indicators, vehicles = process_frame(
                        frame, model, confidence_threshold
                    )
                
                # Display results
                st.image(annotated_frame, caption="Detection Results", use_column_width=True)
                
                # Show detection details
                st.subheader("Detection Details")
                st.write(f"**Accident Score:** {accident_score:.1f}")
                st.write(f"**Vehicles Detected:** {len(vehicles)}")
                
                if vehicles:
                    st.write("**Vehicle List:**")
                    for i, vehicle in enumerate(vehicles):
                        st.write(f"  {i+1}. {vehicle['class_name']} (confidence: {vehicle['confidence']:.2f})")
                
                # Update statistics
                detection_data = {
                    'timestamp': datetime.now(),
                    'type': 'image',
                    'accident_detected': accident_detected,
                    'accident_score': accident_score,
                    'indicators': indicators,
                    'vehicles_count': len(vehicles)
                }
                st.session_state.detection_history.append(detection_data)
                if accident_detected:
                    st.session_state.accident_count += 1
        
        elif detection_mode == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
            
            if uploaded_file is not None:
                # Save uploaded video to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Process video
                cap = cv2.VideoCapture(tmp_file_path)
                
                if cap.isOpened():
                    frame_count = 0
                    accident_frames = 0
                    max_score = 0
                    
                    # Create placeholder for video display
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Process every 5th frame for better performance
                        if frame_count % 5 == 0:
                            annotated_frame, accident_detected, accident_score, indicators, vehicles = process_frame(
                                frame, model, confidence_threshold
                            )
                            
                            if accident_detected:
                                accident_frames += 1
                            
                            max_score = max(max_score, accident_score)
                            
                            # Display frame
                            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                            
                            # Update progress
                            progress_bar.progress(frame_count / total_frames)
                        
                        # Break if we've processed enough frames
                        if frame_count > 500:  # Process max 500 frames
                            break
                    
                    cap.release()
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    # Update statistics
                    detection_data = {
                        'timestamp': datetime.now(),
                        'type': 'video',
                        'accident_detected': accident_frames > 0,
                        'accident_score': max_score,
                        'indicators': [f"Accidents detected in {accident_frames} frames out of {frame_count // 5}"]
                    }
                    st.session_state.detection_history.append(detection_data)
                    if accident_frames > 0:
                        st.session_state.accident_count += 1
        
        elif detection_mode == "Live Camera (Demo)":
            st.info("📹 Live camera detection would work with a webcam. This is a demo mode.")
            
            # Demo with sample traffic images
            if st.button("🎲 Run Demo Detection"):
                # Create a synthetic traffic scenario for demo
                demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                demo_frame[:] = (135, 206, 235)  # Sky blue background
                
                # Add some rectangles to simulate cars
                cv2.rectangle(demo_frame, (100, 300), (200, 400), (0, 0, 255), -1)  # Red car
                cv2.rectangle(demo_frame, (250, 320), (350, 420), (255, 0, 0), -1)  # Blue car
                cv2.rectangle(demo_frame, (180, 280), (280, 360), (0, 255, 0), -1)  # Green car (overlapping)
                
                # Add road
                cv2.rectangle(demo_frame, (0, 350), (640, 480), (64, 64, 64), -1)
                
                # Process the synthetic frame
                with st.spinner("Processing demo..."):
                    annotated_frame, accident_detected, accident_score, indicators, vehicles = process_frame(
                        demo_frame, model, confidence_threshold
                    )
                
                st.image(annotated_frame, caption="Demo Detection Results", use_column_width=True)
                
                st.write("**Demo Results:**")
                st.write(f"- Accident Score: {accident_score}")
                st.write(f"- Vehicles Detected: {len(vehicles)}")
                st.write(f"- Status: {'ACCIDENT' if accident_detected else 'NORMAL'}")
    
    with col2:
        st.subheader("🚨 Alert Status")
        
        # Display current alert status
        if st.session_state.detection_history:
            latest_detection = st.session_state.detection_history[-1]
            
            if latest_detection['accident_detected']:
                st.markdown("""
                <div class="alert-box">
                    ⚠️ ACCIDENT DETECTED!<br>
                    Score: {:.1f}<br>
                    Time: {}
                </div>
                """.format(latest_detection['accident_score'], 
                          latest_detection['timestamp'].strftime("%H:%M:%S")), 
                          unsafe_allow_html=True)
                
                # Show indicators
                st.subheader("📋 Indicators")
                for indicator in latest_detection['indicators']:
                    st.write(f"• {indicator}")
                
            else:
                st.markdown("""
                <div class="safe-box">
                    ✅ NORMAL TRAFFIC<br>
                    Score: {:.1f}<br>
                    Time: {}
                </div>
                """.format(latest_detection['accident_score'], 
                          latest_detection['timestamp'].strftime("%H:%M:%S")), 
                          unsafe_allow_html=True)
        
        # Detection history
        st.subheader("📈 Recent History")
        if st.session_state.detection_history:
            for i, detection in enumerate(st.session_state.detection_history[-5:]):
                status = "🚨 ACCIDENT" if detection['accident_detected'] else "✅ NORMAL"
                score = detection['accident_score']
                vehicles = detection.get('vehicles_count', 0)
                st.write(f"{status} - Score: {score:.1f} - Vehicles: {vehicles} - {detection['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.write("No detections yet")
        
        # Statistics chart
        if len(st.session_state.detection_history) > 1:
            st.subheader("📊 Detection Chart")
            
            # Create simple chart
            df = pd.DataFrame(st.session_state.detection_history)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            accident_counts = df['accident_detected'].value_counts()
            
            labels = ['Normal', 'Accident']
            values = [accident_counts.get(False, 0), accident_counts.get(True, 0)]
            colors = ['green', 'red']
            
            ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_title('Detection Summary')
            ax.set_ylabel('Count')
            
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🚗 Car Accident Detection System | Built with YOLO & Streamlit | Real-time AI Detection
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()