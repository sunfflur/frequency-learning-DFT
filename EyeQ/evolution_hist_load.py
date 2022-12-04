import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn


#### EVOLUTION CURVES PLOT BY HIST_LOAD ###
def evolution_hist_load(history, language='pt-br'):
    if language=='en':
        plt.figure(figsize=(6,4))
        plt.plot(history['loss'], 'r-', label='Training loss')
        plt.plot(history['val_loss'], 'r--', label='Validation loss')
        plt.plot(history['accuracy'], 'g-', label = 'Training accuracy')
        plt.plot(history['val_accuracy'], 'g--', label = 'Validation accuracy')
        plt.xlabel('Epoch', fontsize=14), plt.ylabel('Magnitude', fontsize=14)
        plt.title('Loss and Accuracy evolution', fontsize=14)
        plt.legend(loc='best')
        plt.show()
    elif language=='pt-br':
        plt.figure(figsize=(6,4))
        plt.plot(history['loss'], 'r-', label='Perda de treinamento')
        plt.plot(history['val_loss'], 'r--', label='Perda de validação')
        plt.plot(history['accuracy'], 'g-', label = 'Acurácia de treinamento')
        plt.plot(history['val_accuracy'], 'g--', label = 'Acurácia de validação')
        plt.xlabel('Época', fontsize=14), plt.ylabel('Magnitude', fontsize=14)
        plt.title('Evolução de perda e acurácia', fontsize=14)
        plt.legend(loc='best')
        plt.show()
