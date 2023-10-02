import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# Create a FaceAnalysis instance for both face detection and recognition with specified input size
recognition = FaceAnalysis(name='buffalo_l')
recognition.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))  

# Load the images and store the embeddings and names in the dictionary
embeddings = {}
image_paths = ['Passport.jpg']
names = ['Jane Doe']

for image_path, name in zip(image_paths, names):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 600))
    # Detect faces and calculate embeddings using the 'buffalo_l' model
    faces= recognition.get(image)

    for idx, face in enumerate(faces):
        # Calculate the embedding
        face_embedding = face.embedding

        if name not in embeddings:
            embeddings[name] = []

        embeddings[name].append(face_embedding)

