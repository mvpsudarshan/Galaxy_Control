import streamlit as st
# Do not import cv2 or mediapipe at the very top level
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class GalaxyTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Lazy loading: Import inside the function 
        # This prevents the app from crashing on startup if dependencies are slow
        import cv2
        import mediapipe as mp
        
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Initialize hands here
        if not hasattr(self, 'hands'):
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7
            )
            
        # ... your existing galaxy logic ...
        return img
