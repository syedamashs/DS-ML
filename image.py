import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=(64,64),
    batch_size=10,
    class_mode = 'categorical'
)

test_data = datagen.flow_from_directory(
    "dataset/test",
    target_size=(64,64),
    batch_size=10,
    class_mode='categorical'
)


cnn = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(train_data.num_classes,activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(train_data,epochs=1)

ans = cnn.evaluate(test_data)[1]
print(ans)


img_path = '1.jpg'

img = image.load_img(img_path, target_size=(64,64))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array,axis=0)

prediction = cnn.predict(img_array)

classes = list(train_data.class_indices.keys())
predicted_classes = classes[np.argmax(prediction)]

print(predicted_classes)


from sklearn.neighbors import KNeighborsClassifier

X_train_list = []
y_train_list = []

for i in range(len(train_data)):
    X, y = train_data[i]
    X_train_list.append(X)
    y_train_list.append(y)

X_train = np.vstack(X_train_list)
y_train = np.vstack(y_train_list)


X_test_list = []
y_test_list = []

for i in range(len(test_data)):
    X, y = test_data[i]
    X_test_list.append(X)
    y_test_list.append(y)

X_test = np.vstack(X_test_list)
y_test = np.vstack(y_test_list)



X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(X_test),-1)

y_train = np.argmax(y_train,axis=1)
y_test = np.argmax(y_test,axis=1)



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

knn_acc = knn.score(X_test,y_test)
print(knn_acc)

