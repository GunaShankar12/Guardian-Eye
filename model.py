from retinaface import RetinaFace
import time
import cv2
import matplotlib.pyplot as plt
img_path = './images/ds_test1.jpg'

start_time = time.time()
img = cv2.imread(img_path)
'''
output for one face
{
    'score': 0.9994959831237793,
    'facial_area': [200, 238, 221, 264],
    'landmarks': 
        {
            'right_eye': [207.93968, 247.9548],
            'left_eye': [217.21861, 248.2999],
            'nose': [213.07979, 252.90683], 
            'mouth_right': [208.32773, 257.35257], 
            'mouth_left': [216.09364, 257.66913]
        }
}
'''
obj = RetinaFace.detect_faces(img_path)
end_time = time.time()
print("Detection Time: ", end_time - start_time)


for key in obj.keys():
    identity = obj[key]
    # print(identity)
    facial_area = identity['facial_area']
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (0, 255, 0), 2)
    break
plt.imshow(img[:,:,::-1])
plt.show()

plt.imsave("./images/ds_test1_result.jpg", img[:,:,::-1])


# face recognition
from deepface import DeepFace
img_base = "./images/ds_base.jpg"
img_test = "./images/ds_test1.jpg"
start_time = time.time()
result = DeepFace.verify(img1_path = img_base, img2_path = img_test, model_name = 'Facenet', distance_metric = 'euclidean_l2')
end_time = time.time()
print("Recoginition Time: ", end_time - start_time)
print("result", result)