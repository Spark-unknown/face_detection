import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class GestureAndFaceRecognizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Gesture and Face Recognizer")
        
        self.video_capture = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()
        
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        self.gesture_label = tk.Label(self.master, text="Gesture: None", font=("Arial", 24))
        self.gesture_label.pack()
        
        self.face_label = tk.Label(self.master, text="Face: Not Detected", font=("Arial", 24))
        self.face_label.pack()
        
        self.prev_hand_landmarks = None
        self.update()
    
    def recognize_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Basic finger counting
        fingers_up = sum([
            thumb_tip.y < wrist.y,
            index_tip.y < middle_tip.y,
            middle_tip.y < wrist.y,
            ring_tip.y < wrist.y,
            pinky_tip.y < wrist.y
        ])
        
        if fingers_up == 1 and index_tip.y < middle_tip.y:
            return "One"
        elif fingers_up == 2 and index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y:
            return "Two"
        elif fingers_up == 3 and index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y < wrist.y:
            return "Three"
        elif fingers_up == 4 and thumb_tip.y > index_tip.y:
            return "Four"
        elif fingers_up == 5:
            return "Five"
        elif self.is_waving(hand_landmarks):
            return "Waving"
        elif self.is_swiping(hand_landmarks):
            return "Swiping"
        else:
            return "Unknown"
    
    def is_waving(self, hand_landmarks):
        if self.prev_hand_landmarks is None:
            self.prev_hand_landmarks = hand_landmarks
            return False
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        prev_wrist = self.prev_hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        dx = wrist.x - prev_wrist.x
        dy = wrist.y - prev_wrist.y
        
        self.prev_hand_landmarks = hand_landmarks
        
        return abs(dx) > 0.02 and abs(dy) < 0.01
    
    def is_swiping(self, hand_landmarks):
        if self.prev_hand_landmarks is None:
            self.prev_hand_landmarks = hand_landmarks
            return False
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        prev_wrist = self.prev_hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        dx = wrist.x - prev_wrist.x
        
        self.prev_hand_landmarks = hand_landmarks
        
        return abs(dx) > 0.04
    
    def update(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.recognize_gesture(hand_landmarks)
                    self.gesture_label.config(text=f"Gesture: {gesture}")
            else:
                self.gesture_label.config(text="Gesture: None")
                self.prev_hand_landmarks = None
            
            # Face detection
            face_results = self.face_detection.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    self.mp_drawing.draw_detection(frame, detection)
                self.face_label.config(text="Face: Detected")
            else:
                self.face_label.config(text="Face: Not Detected")
            
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
        
        self.master.after(10, self.update)

    def __del__(self):
        self.video_capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureAndFaceRecognizer(root)
    root.mainloop()