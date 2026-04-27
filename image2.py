import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_data(folder_path):

    X=[]
    y=[]

    classes = os.listdir(folder_path)

    for label, class_name in enumerate(classes):
        class_path = os.path.join(folder_path,class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path,img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (64,64))
            img = img/255.0

            X.append(img.flatten())
            y.append(label)

    return np.array(X), np.array(y), classes



X_train, y_train, classes = load_data("dataset/train")
X_test, y_test, _ = load_data("dataset/test")

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

print(rf.score(X_test,y_test))



img_path = input("\nEnter image path: ")

img = cv2.imread(img_path)

if img is None:
    print("Invalid image path!")
else:
    img = cv2.resize(img, (64,64))
    img = img / 255.0

    img = img.flatten().reshape(1, -1)

    pred = rf.predict(img)

    print("Predicted Class:", classes[pred[0]])
