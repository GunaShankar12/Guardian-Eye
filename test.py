import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# Create a FaceAnalysis instance for both face detection and recognition with specified input size
recognition = FaceAnalysis(name='buffalo_l')
recognition.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

# Load the new image for face recognition
new_image = cv2.imread('Passport-min.jpg')
new_image = cv2.resize(new_image, (600, 600))

# Detect faces and calculate embeddings using the 'buffalo_l' model in the new image
new_faces= recognition.get(new_image)
new_embeddings = []

for new_face in new_faces:
    # Calculate the embedding
    new_face_embedding = new_face.embedding
    new_embeddings.append(new_face_embedding)

# Compare the new embeddings with the saved embeddings and draw bounding boxes around matched faces
image_with_boxes = new_image.copy()


for i, new_embedding in enumerate(new_embeddings):
    match_found = False

    for name, reference_embeddings in embeddings.items():
        for reference_embedding in reference_embeddings:
            distance = cosine_similarity([new_embedding], [reference_embedding])

            print(distance)
            # If the distance is less than a threshold, identify the person in the new image
            if (distance) > 0.4:
                match_found = True
                break

        if match_found:
            break

    if match_found:
        x, y, width, height = new_faces[i].bbox.astype(int)
        # print(width,height)
        cv2.rectangle(image_with_boxes, (x, y), ((width), (height)), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print('No match found for face', i)

# Display the image with bounding boxes
image_with_boxes = cv2.resize(image_with_boxes, (600, 600))
cv2.imshow("Recognized Faces", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
