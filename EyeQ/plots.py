import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MultipleLocator


def my_violin_plot(data_to_plot, labels=['A', 'B'], xlabel='Sample name', ylabel='Observed values', title='Title', cm='Pastel2_r',
                   savefig_path='G:\\Meu Drive\\Mestrado\\Experimentos\\exp2-KTD\\exp2-pynb\\exp2-DHT\\exp_02-DHT-tests\\optimizers\\violin_plot'):

    # Create a figure instance
    fig = plt.figure(figsize=(9,4))
    
    quartile1, medians, quartile3 = np.percentile(data_to_plot, [25, 50, 75], axis=1) #axis=0
    inds = np.arange(1, len(medians) + 1)

    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xticklabels(labels)

    ax.scatter(inds, medians, marker='o', color='white', s=20, zorder=3)

    # Create the boxplot
    bp = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)

    heights = [violin.get_paths()[0].get_extents().height for violin in bp['bodies']]
    norm = plt.Normalize(min(heights), max(heights))
    cmap = plt.get_cmap(cm)


    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        vp = bp[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(0.5)

    # Make the violin body a cmap with black borders
    for violin, height in zip(bp['bodies'], heights):
         violin.set_color(cmap(norm(height)))
         violin.set_edgecolor('gray')
         violin.set_alpha(1.0)
         violin.set_linewidth(0.5)

    #plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), alpha=violin.get_alpha(), ax=ax) #label='Violin Extent'
    #plt.tight_layout()
    plt.savefig(savefig_path)
    plt.show()


### TRAIN/VALIDATION EVOLUTION PLOT ###

def evolution_curves_plot(history, language='pt-br'):
    if language=='en':
        plt.figure(figsize=(6,4))
        plt.plot(history.history['loss'], 'r-', label='Training loss')
        plt.plot(history.history['val_loss'], 'r--', label='Validation loss')
        plt.plot(history.history['accuracy'], 'g-', label = 'Training accuracy')
        plt.plot(history.history['val_accuracy'], 'g--', label = 'Validation accuracy')
        plt.xlabel('Epoch', fontsize=14), plt.ylabel('Magnitude', fontsize=14)
        plt.title('Loss and Accuracy evolution', fontsize=14)
        plt.legend(loc='best')
        plt.show()
    elif language=='pt-br':
        plt.figure(figsize=(6,4))
        plt.plot(history.history['loss'], 'r-', label='Perda de treinamento')
        #plt.plot(history.history['val_loss'], 'r--', label='Perda de validação')
        plt.plot(history.history['accuracy'], 'g-', label = 'Acurácia de treinamento')
        #plt.plot(history.history['val_accuracy'], 'g--', label = 'Acurácia de validação')
        plt.xlabel('Época', fontsize=14), plt.ylabel('Magnitude', fontsize=14)
        plt.title('Evolução de perda e acurácia', fontsize=14)
        plt.legend(loc='best')
        plt.show()

### CONFUSION MATRIX PLOT ###

def confusion_matrix_plot(data, y_test, model, language='pt-br'): 
    y_pred = model.predict(data)
    plt.figure(figsize = (12,8))
    cm = confusion_matrix(tf.argmax(y_test, axis=1), tf.argmax(y_pred, axis=1))
    acc = np.trace(cm)/data.shape[0] * 100
    erros = tf.reduce_sum(cm)-tf.linalg.trace(cm)
    if language=='en':
        plt.title('Test set - accuracy %.2f%% / %d incorrect classifications' % (acc, erros), fontsize=14)
    elif language=='pt-br':
        plt.title('Conjunto de teste - %.2f%% de acurácia / %d classificações incorretas' % (acc, erros), fontsize=14)
    labels = ['good','reject']
    sn.heatmap(cm, cmap='Pastel1_r', linewidths=.1, annot=True, fmt=".6g", xticklabels=labels, yticklabels=labels)
    plt.xlabel('y_pred',fontsize=14)
    plt.ylabel('y_test',fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

### INCORRECT CLASSIFICATION EXAMPLES PLOT ###

def incorrect_class_plot(data, x_test, y_test, model):
    dic = {0:'blanket1',1:'blanket2',2:'canvas1',3:'ceiling1',4:'ceiling2',5:'cushion1',
           6:'floor1',7:'floor2',8:'grass1',9:'lentils1',10:'linseeds1',11:'oatmeal1',
           12:'pearlsugar1',13:'rice1',14:'rice2',15:'rug1',16:'sand1',17:'scarf1',
           18:'scarf2',19:'screen1',20:'seat1',21:'seat2',22:'sesameseeds1',23:'stone1',
           24:'stone2',25:'stone3',26:'stoneslab1',27:'wall1'}

    y_pred = model.predict(data)
    dif = tf.argmax(y_test, axis=1)-tf.argmax(y_pred, axis=1)
    indices = np.where(dif!=0)[0]
    plt.figure(figsize=(14,6))
    plt.subplot(121), plt.imshow(tf.squeeze(x_test[indices[0]]), cmap='gray')
    r1 = tf.argmax(y_test, axis=1)[indices[0]]
    f1 = tf.argmax(y_pred, axis=1)[indices[0]]
    p1=np.amax((y_pred)[indices[0]])

    #t1 = "Actual class = {}\nPredicted class = {}\nPrediction probability = {:.2f} %" \
    #            .format(dic[int(r1)], dic[int(f1)], p1*100)
    t1 = "Classe verdadeira = {}\nClasse predita = {}\nProbabilidade de predição = {:.2f} %" \
                .format(dic[int(r1)], dic[int(f1)], p1*100)
    #plt.title('Classe: %s - Predição: %s' % (dic[int(r1)], dic[int(f1)]))
    #plt.title('Class: %s - Prediction: %s' % (dic[int(r1)], dic[int(f1)]))
    plt.title(t1,fontsize=14)

    p2=np.amax((y_pred)[indices[1]])
    plt.subplot(122), plt.imshow(tf.squeeze(x_test[indices[1]]), cmap='gray')
    r2 = tf.argmax(y_test, axis=1)[indices[1]]
    f2 = tf.argmax(y_pred, axis=1)[indices[1]]
    #plt.title('Classe: %s - Predição: %s' % (dic[int(r2)], dic[int(f2)]))
    #plt.title('Class: %s - Prediction: %s' % (dic[int(r2)], dic[int(f2)]))
    #t2 = "Actual class = {}\nPredicted class = {}\nPrediction probability = {:.2f} %" \
    #            .format(dic[int(r2)], dic[int(f2)], p2*100)
    t2 = "Classe verdadeira = {}\nClasse predita = {}\nProbabilidade de predição = {:.2f} %" \
                .format(dic[int(r2)], dic[int(f2)], p2*100)
    plt.title(t2,fontsize=14)
    #plt.subplot(133), plt.imshow(tf.squeeze(x_test[indices[2]]), cmap='gray')
    #r3 = tf.argmax(y_test, axis=1)[indices[2]]
    #f3 = tf.argmax(y_pred, axis=1)[indices[2]]
    #plt.title('Classe: %s - Predição: %s' % (dic[int(r3)], dic[int(f3)]))
    plt.show()

### FREQUENCY LAYER WEIGHTS PLOT ###

def freq_weights_plot(model):
    plt.figure(figsize=(9,3))
    plt.plot(model.get_weights()[0])
    #plt.plot(model.get_weights()[1])
    #plt.plot(model.get_weights()[2])
    #plt.plot(model.get_weights()[3])
    plt.title('Camada em frequência',fontsize=14)
    #plt.title('Frequency Layer')
    plt.xlabel('Raio (r)',fontsize=14), plt.ylabel('Pesos',fontsize=14)
    #plt.xlabel('Radius (r)'), plt.ylabel('Weights')
    plt.show()
