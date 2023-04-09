from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import os

from Firebase import extraccion

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        figura = "Dodecaedro"
    elif answer == 1:
        figura = "Hexaedro"
    elif answer == 2:
        figura = "Octaedro"
    elif answer == 3:
        figura = 'Piramide Cuadrangular'
    elif answer == 4:
        figura = 'Piramide Triangular'
    elif answer == 5:
        figura = 'Prisma Triangular'
    elif answer == 6:
        figura = 'Prisma Cuadrangular'
    elif answer == 7:
        figura = 'Tetraedro'

    return figura

extraccion()
carpeta = './images'
contenido = os.listdir(carpeta)

for imagenes in contenido:
    img_path = carpeta + '/' + imagenes
    print(img_path)
    figura = predict(img_path)
    print('Predicciones: ' + figura + '\n')