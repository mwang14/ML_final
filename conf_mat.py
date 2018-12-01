from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_cm(conf_matrix, class_names, normalized=False):
    if normalized == True:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print('Normalized')
    else:
        print('Not Normalized')

    plt.clf()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalized == True:
        formatting = '.2f'
    else:
        formatting = 'd'

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            num = conf_matrix[i][j]
            plt.text(j, i, format(num, formatting), horizontalalignment='center')
    plt.show()
