# -*- coding: utf-8 -*-
"""Projeto_Reconhecimento_Facial_OpenCV_LBPH.ipynb

## Implementando Reconhecimento Facial com OpenCV e LBPH
Diego Hartwig - 2024
"""

# Importação das bibliotecas
from PIL import Image
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import drive
import zipfile
import os

# Montando o drive
drive.mount('/content/drive')

#!rm -rf 'yalefaces'

path = '/content/drive/MyDrive/Projetos_IA/Reconhecimento_Facial_OpenCV/yalefaces.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

"""# Pré-processamento das imagens"""

print(os.listdir('/content/yalefaces/train'))

def get_image_data():
  paths = [os.path.join('/content/yalefaces/train', f) for f in os.listdir('/content/yalefaces/train')]
  print(paths)
  faces = []
  ids = []

  for path in paths:
    imagem = Image.open(path).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    #print(id)
    ids.append(id)
    faces.append(imagem_np)
  return np.array(ids), faces

ids, faces = get_image_data()

"""# Treinamento do classificador LBPH"""

lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=4, neighbors=14, grid_x=9, grid_y=9)
lbph_classifier.train(faces, ids)
lbph_classifier.write('classificadorLBPH.yml')

"""# Reconhecimento de faces"""

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('classificadorLBPH.yml')

imagem_teste = '/content/yalefaces/test/subject07.happy.gif'

imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')
imagem_np

previsao = lbph_face_classifier.predict(imagem_np)
previsao

saida_esperada = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
saida_esperada

cv2.putText(imagem_np, 'Pred: ' + str(previsao[0]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
cv2.putText(imagem_np, 'Exp: ' + str(saida_esperada), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
cv2_imshow(imagem_np)

"""# Avaliação do classificador"""

paths = [os.path.join('/content/yalefaces/test', f) for f in os.listdir('/content/yalefaces/test')]
previsoes = []
saidas_esperadas = []

for path in paths:
  imagem = Image.open(path).convert('L')
  imagem_np = np.array(imagem, 'uint8')
  previsao, _ = lbph_face_classifier.predict(imagem_np)
  saida_esperada = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

  previsoes.append(previsao)
  saidas_esperadas.append(saida_esperada)

type(previsoes), type(saidas_esperadas)

previsoes = np.array(previsoes)
saidas_esperadas = np.array(saidas_esperadas)

type(previsoes), type(saidas_esperadas)

previsoes

saidas_esperadas

from sklearn.metrics import accuracy_score
accuracy_score (saidas_esperadas, previsoes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(saidas_esperadas, previsoes)
cm

import seaborn
seaborn.heatmap(cm, annot=True)