from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizacja wartości pikseli do zakresu [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_test_original = y_test

# Konwersja etykiet na format one-hot encoding
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Definicja nazw klas
class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń', 'pies', 'żaba', 'koń', 'statek', 'ciężarówka']

if __name__ == "__main__":
    # Wyświetlenie kilku przykładowych obrazów
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        plt.xlabel(class_names[np.argmax(y_train[i])])
    plt.show()