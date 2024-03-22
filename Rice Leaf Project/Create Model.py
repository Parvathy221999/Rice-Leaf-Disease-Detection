import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255,fill_mode='nearest')
batch_size = 32
total_image = datagen.flow_from_directory(r"C:\Users\parva\Music\Full Data",
    target_size=(220, 220),
    batch_size=batch_size,
    class_mode='categorical'
)

training_image = datagen.flow_from_directory(r"C:\Users\parva\Music\Training Data",
    target_size=(220, 220),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_image = datagen.flow_from_directory(r"C:\Users\parva\Music\Validation Data",
    target_size=(220, 220),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(total_image.class_indices)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(220,220,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(150, kernel_size=(3, 3), activation='relu',name='conv_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
model.fit(total_image,epochs=20,validation_data=validation_image,batch_size=500)


test_loss, test_accuracy = model.evaluate(validation_image)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
img_path = r"C:\Users\parva\Music\Full Data\Bacterial leaf blight\DSC_0398.JPG"
img = image.load_img(img_path, target_size=(220, 220))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array/255.0
prediction = model.predict(img_array)


class_label = np.argmax(prediction)
class_names = ["Bacterial leaf blight", 'Brown spot', 'Leaf smut',"unrelated_image"]
predicted_class = class_names[class_label]
#plt.imshow(img)
#plt.title('Predicted class: ' +  predicted_class)
#plt.show()

model.save("rice_leaf_diseases_classification_model.h5")
