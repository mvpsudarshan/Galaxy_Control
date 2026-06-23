import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
import math

# Session State for persistence
if 'active' not in st.session_state: st.session_state.active = True
if 'direction' not in st.session_state: st.session_state.direction = 1

class GalaxyTransformer(VideoTransformerBase):
    def __init__(self):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0].landmark
            thumb, index, pinky = hand[4], hand[8], hand[20]
            
            # Gesture Logic
            is_curled = all(math.hypot(hand[i].x - hand[0].x, hand[i].y - hand[0].y) < 0.15 for i in [12, 16, 20])
            dist_thumb_index = math.hypot(thumb.x - index.x, thumb.y - index.y)
            
            # Master Toggle
            if is_curled and dist_thumb_index > 0.15: 
                st.session_state.active = not st.session_state.active
            # Change Direction
            elif math.hypot(thumb.x - pinky.x, thumb.y - pinky.y) < 0.08: 
                st.session_state.direction *= -1

        # HUD Aesthetics
        cv2.putText(img, f"PARTICLES: {'ACTIVE' if st.session_state.active else 'ABSENT'}", (20, 40), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(img, f"DIR: {'CW' if st.session_state.direction==1 else 'CCW'}", (20, 80), 1, 1.5, (255, 255, 0), 2)
        return img

st.title("Galaxy Control Interface")
webrtc_streamer(key="galaxy", video_transformer_factory=GalaxyTransformer)
st.write("Grant camera access to start.")
