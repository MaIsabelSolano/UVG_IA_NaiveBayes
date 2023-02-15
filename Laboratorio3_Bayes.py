'''
# Laboratorio 3: Naive Bayes
Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la Computación 
Inteligencia Artificial

## Integrantes
* Christopher García
* Gabriel Vicente
* Ma. Isabel Solano 
'''

# Librerías
import pandas as pd
import numpy as np
import matplotlib as plt
import string

#Imports necesarios para la task
import re
import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Librerias para limpiar
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Task 1.1
def remove_punctuation ( text ):
  return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

def remove_non_ascii(string):
  return ''.join(char for char in string if ord(char) < 128)

eliminarOtros = lambda str: "".join(re.findall("[\w\s]", str))

Dataset = []
LineasEtiquetadas = []

"""Se realiza la limpieza de datos, eliminando todos los caractéres especiales y de puntuación."""

with open('entrenamiento.txt') as file:
    for line in file:
        temp = remove_non_ascii(eliminarOtros(remove_punctuation(line))).upper()
        Dataset.append(temp)
        
with open('Dataset.txt', 'w') as file:
    for line in Dataset:
        file.write(line)

datos = pd.read_csv('Dataset.txt', sep='\t', header=None, names = ['Label', 'SMS'] )

# Task 1.2

X = datos.iloc[:, 1]
y = datos.iloc[:, 0]

"""Separación entre los datos de entrenamiento y de prueba"""

# Importación de liberías par separar los datos
from sklearn.model_selection import train_test_split
seed = 100
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.25, random_state = seed)

"""Creación de bancos de palabras"""

# Crear diccionarios para los bancos de palabras y las probabilidades de ser spam o ham
prob_spam = {}
prob_ham = {}

spam_count = 0
ham_count = 0

total_count = 0

for i in range(len(X_entreno)):
  # print(X_entreno.get(i))
  if(X_entreno.get(i)): 
    # Evita los datos null
    total_count += 1

    words = X_entreno.get(i).split(" ")
    
    if y_entreno.get(i) == "SPAM":
      spam_count += 1
      for w in words:
        if w in prob_spam:
          prob_spam[w] += 1
        else:
          prob_spam[w] = 1

    if y_entreno.get(i) == "HAM":
      ham_count += 1
      for w in words:
        if w in prob_ham:
          prob_ham[w] += 1
        else:
          prob_ham[w] = 1

"""Aplicación de suavizador de Laplace"""

# k a utilizar
k = 5

# spam
for w in prob_spam:
  prob_spam[w] = (prob_spam[w] + k) / (spam_count + 2 + k)

# ham
for w in prob_ham:
  prob_ham[w] = (prob_ham[w] + k) / (ham_count + 2 + k)

"""Cálculo de probabilidad a priori"""

p_spam = spam_count / total_count
p_ham = ham_count / total_count

print('\nP(HAM) = ', p_ham)
print('P(SPAM) = ', p_spam)

"""Clasificación de mensajes """

# Método de clasificación
def classify(message):
  words = message.split(" ")
  probabilidad_spam = np.log(p_spam)
  probabilidad_ham = np.log(p_ham)

  for w in words:
    # Revisa la probabilidad de ser spam
    if w in prob_spam:
      probabilidad_spam += np.log(prob_spam[w])

    # Revisa la probabilidad de ser ham
    if w in prob_ham:
      probabilidad_ham += np.log(prob_ham[w])

    return "SPAM" if probabilidad_spam >= probabilidad_ham else "HAM"

"""Evaluación del modelo"""

correct = 0

cant_prueba = 0

for i in range(len(X_prueba)):
  if X_prueba.get(i):
    # Evita datos vacíos
    cant_prueba += 1

    if (classify(X_prueba.get(i)) ==  y_prueba.get(i)):
      correct += 1
      # print(y_prueba.get(i))

accuracy = correct / cant_prueba

print("Accuracy: ",accuracy)
print()

# Task 1.3

# Ingreso del string a evaluar
print('A continuación ingrese el SMS a evaluar (preferiblemente en inglés):')
SMS = input('Mensaje: ').upper()
SMS = remove_punctuation(SMS)
SMS = remove_non_ascii(SMS )

print(SMS)
print(classify(SMS))

# Task 1.4
#Dataset y limpieza
DataSetEntrenamiento = pd.read_csv('entrenamiento.txt', sep='\t', names=['clasification', 'texto'])
lemmatizer = WordNetLemmatizer()

#Limpieza con nltk
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]
    return " ".join(words)
DataSetEntrenamiento['texto'] = DataSetEntrenamiento['texto'].apply(preprocess)

#Entrenamiento de la red de bayes con skLearn
X_train, X_test, y_train, y_test = train_test_split(DataSetEntrenamiento['texto'], DataSetEntrenamiento['clasification'], test_size=0.2)
model = Pipeline([('vectorizer', CountVectorizer()),('classifier', MultinomialNB())])
model.fit(X_train, y_train)

#Impresion de resultados
print('Precision: ', model.score(X_test, y_test))
print()
# Clasificacion desde consola
while True:
    texto = input("A continuación ingrese el SMS a evaluar (preferiblemente en inglés) (ENTER para terminar):")
    if texto == "":
        break
    prediction = model.predict([preprocess(texto)])
    print(prediction)

"""

*   ¿Cuál implementación lo hizo mejor? ¿Su implementación o la de la librería?
    * La de la librería sklearn.
    
*   ¿Por qué cree que se debe esta diferencia?
    * Primero que nada podemos decir que la librería cuenta con una precisión mucho mayor
      al momento de realizar cálculos por lo que se esperaba que la librería tuviera mejor rendimiento. 
      Otro punto que consideramos importante mencionar es que la limpieza de dato para 
      la parte 1.4 se realizó con una librería dedicada a esto por lo que se pudieron 
      eliminar palabras/números/carácteres que no habíamos considerado como importantes 
      para poder eliminarlos y finalmente consideramos que la división de datos para entreno, prueba
      y validación pudieron variar haciendo que nuestro modelo no fuera el más exacto.
"""
