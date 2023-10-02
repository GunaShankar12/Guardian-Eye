import cv2
from insightface.app import FaceAnalysis
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

recognition = FaceAnalysis(name='buffalo_l')
recognition.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
data = pd.read_csv('data.csv')
required = data.iloc[:, 1:].values
THRESHOLD = 0.4


def detect_faces(frame):
    faces= recognition.get(frame)
    face_landmarks = []
    for face in faces:
        face_landmarks.append((face.embedding, face.bbox)) # face is a dictonary with keys:['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding']
    return face_landmarks


def compare_embeddings(new_embedding):
    for embedding in required:
        label = embedding[0]
        embedding = embedding[1:]
        s = cosine_similarity([embedding], [new_embedding])
        if(s >= THRESHOLD):
            print(s, label)
            return label
    return None



def webcam():
    video_capture = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = video_capture.read()
            face_landmarks = detect_faces(frame)
            for face_landmark, bbox in face_landmarks:
                tag = compare_embeddings(face_landmark)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                if(tag is not None):
                    cv2.putText(frame, tag, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass


def static_recognition(img_path):
    frame = cv2.imread(img_path)
    face_landmarks = detect_faces(frame)
    for face_landmark, bbox in face_landmarks:
        tag = compare_embeddings(face_landmark)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        if(tag is not None):
            cv2.putText(frame, tag, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Capture', frame)
    cv2.waitKey(0)

'''
to run without warnings: 
python -W ignore <filename>.py
'''
if __name__ == '__main__':
    # static_recognition('./images/group.png')
    webcam()