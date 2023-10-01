import cv2

# Load the image
image = cv2.imread("./images/group2.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the face detector
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_detector.detectMultiScale(gray_image)

# Draw a rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
