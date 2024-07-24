import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import math
import paho.mqtt.publish as publish


NUM_FACE = 1
thresh = 0.2  # You need to define the threshold value for eye aspect ratio
flag = 0  # Initialize the flag for drowsiness detection

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(1)  # Use 0 for default camera

class FaceLandMarks():
    def __init__(self, staticMode=False, maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, 
                                                 max_num_faces=self.maxFace, 
                                                 min_detection_confidence=self.minDetectionCon, 
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.right_eye = np.array([[33, 133], [160, 144], [159, 145], [158, 153]])  # right eye landmark positions
        self.left_eye = np.array([[263, 362], [387, 373], [386, 374], [385, 380]])  # left eye landmark positions

    def eye_feature(self, landmarks):
        ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
        :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
        :return: Eye feature value
        '''
        return (self.eye_aspect_ratio(landmarks, self.left_eye) + self.eye_aspect_ratio(landmarks, self.right_eye)) / 2

    def distance(self, p1, p2):
        ''' Calculate distance between two points
        :param p1: First Point 
        :param p2: Second Point
        :return: Euclidean distance between the points. (Using only the x and y coordinates).
        '''
        return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5

    def eye_aspect_ratio(self, landmarks, eye):
        ''' Calculate the ratio of the eye length to eye width. 
        :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
        :param eye: List containing positions which correspond to the eye
        :return: Eye aspect ratio value
        '''
        N1 = self.distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
        N2 = self.distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
        N3 = self.distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
        D = self.distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
        return (N1 + N2 + N3) / (3 * D)

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        ear = None

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                ear = self.eye_feature(np.array(face))
                faces.append(face)

        return img, faces, ear


detector = FaceLandMarks()
status = False
last_status = False
while run:
    _, frame = camera.read()


    if frame is None:
        st.write('Camera not found')
        break

    img, faces, ear = detector.findFaceLandmark(frame)

    if ear is not None and ear < thresh:
        # print(ear)
        flag += 1
        if flag >= 15:
            cv2.putText(img, "Siswa Mengantuk!!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            status = True
        else:
            cv2.putText(img, "Siswa Konstentrasi!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            status = False
    else:
        cv2.putText(img, "Siswa Konstentrasi!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        flag =0
        status = False

    if status != last_status:
        last_status = status
        if status:
            publish.single("/sic5/kelompok14", "1", hostname="broker.hivemq.com")

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

camera.release()
