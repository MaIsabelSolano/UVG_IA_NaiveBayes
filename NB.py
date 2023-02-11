#Clase Naive Bayes

# Librerías
import pandas as pd
import numpy as np
import matplotlib as plt

class NB(): 
    def __init__(self, data_entrenamiento):
        
        # Lectura de los datos
        datos = pd.read_csv(data_entrenamiento, sep='\t', header=None)
        print(datos.head())

        # Creación del data frame

        # Limpieza de los datos 
        pass

nb = NB('entrenamiento.txt')