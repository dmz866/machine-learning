import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

#Imagenes de numeros en 8 bits
digitos = load_digits()

X = digitos.data
X = X / 16.0 # Values are between 0 - 16 for each pixel. Normalize values between 0 and 1.
X_entrena, X_prueba = train_test_split(X, test_size=0.2, random_state=42)

imagen_entrada = Input(shape=(64, ))
codificado = Dense(32, activation='relu') (imagen_entrada)
decodificado = Dense(64, activation='sigmoid') (codificado)

autoencoder = Model(imagen_entrada, decodificado)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_entrena, X_entrena, epochs=100, batch_size=256, shuffle=True, validation_data=(X_prueba, X_prueba))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_prueba[i].reshape(8, 8))

    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(autoencoder.predict(X_prueba)[i].reshape(8, 8))


#plt.imshow(digitos['data'][4].reshape(8,8)) #Number 4 image
#plt.imshow(digitos.images[4]) #Number 4 image
plt.show()