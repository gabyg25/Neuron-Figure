import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model

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
        print("pred: dodecaedro")
    elif answer == 1:
        print("pred: hexaedro")
    elif answer == 2:
        print("pred: octaedro")
    elif answer == 3:
        print('Pred: piramide cuadrangular')
    elif answer == 4:
        print('Pred: piramide triangular')
    elif answer == 5:
        print('Pred: prisma triangular')
    elif answer == 6:
        print('Pred: prisma cuadrangular')
    elif answer == 7:
        print('Pred: tetraedro')

    return answer

predict('./images/CAP564397764.jpg')