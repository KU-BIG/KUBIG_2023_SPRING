import cv2
import os

path_dir = "crolling/cheheyonwook/"
file_list = os.listdir(path_dir)
print(len(file_list))
file_name_list = []
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))

print(file_name_list[1])

def cutting_face_save(img,name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped = img[y:y + h, x:x + w]
        resize =cv2.resize(cropped, (180,180))
        cv2.imwrite(f"crop/{name}.jpg", resize)

for name in file_name_list:
    print(name)
    img = cv2.imread(f"crolling/cheheyonwook/{name}.jpg")
    cutting_face_save(img, name)
