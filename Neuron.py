import os
import tensorflow as tf
from keras.models import Sequential
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns


epocas = 10
img_height = 100  # alto
img_width = 100  # ancho

train_ds = tf.keras.utils.image_dataset_from_directory("./Dataset/Entrenamiento", image_size=(img_height, img_width), batch_size=240)
val_ds = tf.keras.utils.image_dataset_from_directory("./Dataset/Validacion", image_size=(img_height, img_width), batch_size=213)
class_names = train_ds.class_names

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)

num_clases = len(class_names)
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)), 
                                        tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomZoom(0.1),])

modelo = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(3, 3),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(3, 3),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(3, 3),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_clases, activation='softmax')
])

modelo.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


history = modelo.fit(
        train_ds, 
        steps_per_epoch=(len(train_ds)),
        epochs=20,
        validation_data= val_ds,
        validation_steps=len(val_ds))

#Creacion del modelo
target_dir = './modelo/'
if not os.path.exists(target_dir):
        os.mkdir(target_dir)
modelo.save('./modelo/modelo.h5')
modelo.save_weights('./modelo/pesos.h5')

#Grafica de Error de Entrenamiento
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validaciones')
plt.legend(loc='upper right')
plt.show()

# Matrix de Confusion
test_labels = []
test_images = []

for img, labels in val_ds.take(1):
        test_images.append(img)
        test_labels.append(labels)
        imgAux = np.array(img)
        labelsAux = np.array(labels)

y_pred = np.argmax(modelo.predict(test_images), axis=1).flatten()
y_true = np.asarray(test_labels).flatten()
test_acc = sum(y_pred == y_true) / len(y_true)
print(("Test accuracy: {:.2f}%".format(test_acc * 100)))

consfusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(consfusion_matrix.numpy(), xticklabels=class_names,yticklabels=class_names, annot=True, fmt="d")
plt.title('Matriz de confusion')
plt.xlabel('Predicciones')
plt.ylabel('Datos')
plt.show()
