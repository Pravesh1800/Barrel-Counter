import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    return intersection / (box1_area + box2_area - intersection)

def process_video(model, video_file, confidence_threshold, iou_threshold):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a temporary file for the output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    unique_objects = defaultdict(list)
    class_counts = defaultdict(int)
    
    progress_bar = st.progress(0)
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        results = model(frame)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)
                class_name = model.names[cls]
                conf = float(box.conf)
                
                if conf < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                current_box = [x1, y1, x2, y2]
                
                is_new_object = True
                for tracked_box in unique_objects[class_name]:
                    if iou(current_box, tracked_box) > iou_threshold:
                        is_new_object = False
                        break
                
                if is_new_object:
                    unique_objects[class_name].append(current_box)
                    class_counts[class_name] += 1
        
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    return class_counts, output_file.name

st.title('Chemical Barrel Counter')

model_file = st.file_uploader("Upload YOLO model file (.pt)", type=['pt'])

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
        tmp_model_file.write(model_file.read())
        model_path = tmp_model_file.name

    # Load the model
    model = YOLO(model_path)
    st.success("Model loaded successfully!")

    # File uploader for video
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])

    if video_file is not None:
        # Sliders for confidence and IoU thresholds
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)

        if st.button('Process Video'):
            with st.spinner('Processing video...'):
                class_counts, output_video_path = process_video(model, video_file, confidence_threshold, iou_threshold)

            st.success('Video processed successfully!')

            # Display results
            st.subheader('Unique Object Counts:')
            for class_name, count in class_counts.items():
                st.write(f"{class_name}: {count}")


            with open(output_video_path, 'rb') as file:
                st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
else:
    st.warning("Please upload a YOLO model file (.pt) to get started.")