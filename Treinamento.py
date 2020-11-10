
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# Inicializa paramentros
quantidade_treinos = 10
tamanho_lote = 32

# Carregamento das Imagens
caminho_imagens = list(paths.list_images("dataset"))
dados = []
labels = []

# loop sobre os caminhos da imagem
for imageCaminho in caminho_imagens:
	label = imageCaminho.split(os.path.sep)[-2]
	imagem = load_img(imageCaminho, target_size=(224, 224))
	imagem = img_to_array(imagem)
	imagem = preprocess_input(imagem)
	dados.append(imagem)
	labels.append(label)

dados = np.array(dados, dtype="float32")
labels = np.array(labels)

# executar codificação one-hot 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particione os dados em divisões de treinamento e teste usando 80% dos dados para treinamento e os 20% restantes para teste
(trainX, testX, trainY, testY) = train_test_split(dados, labels, test_size=0.20, stratify=labels, random_state=42)

# carregar a rede MobileNetV2, garantindo que os conjuntos de camadas FC principais sejam deixados de lado
# construa a cabeça do modelo que será colocado em cima do modelo base
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

modelo_cabeca = baseModel.output
modelo_cabeca = AveragePooling2D(pool_size=(7, 7))(modelo_cabeca)
modelo_cabeca = Flatten(name="flatten")(modelo_cabeca)
modelo_cabeca = Dense(128, activation="relu")(modelo_cabeca)
modelo_cabeca = Dropout(0.5)(modelo_cabeca)
modelo_cabeca = Dense(2, activation="softmax")(modelo_cabeca)

modelo = Model(inputs=baseModel.input, outputs=modelo_cabeca)

for layer in baseModel.layers:
	layer.trainable = False

# ------------------------ #

print("--> Compilando o modelo.")
modelo.compile(loss="binary_crossentropy",  metrics=["accuracy"])

print("--> Treinando para reconhecer a cabeça.")
cabeca = modelo.fit(trainX, trainY, steps_per_epoch=len(trainX) // tamanho_lote, validation_data=(testX, testY), validation_steps=len(testX) // tamanho_lote, epochs=quantidade_treinos)

# previsões
print("--> avaliação de rede neural.")
predIdxs = modelo.predict(testX, batch_size=tamanho_lote)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# salva modelo
print("--> Salvando o modelo detector de mascara.")
modelo.save("detectores/mask_detector.model", save_format="h5")

# reporte
N = quantidade_treinos
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), cabeca.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), cabeca.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), cabeca.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), cabeca.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
