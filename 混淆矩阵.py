import numpy as np
import matplotlib.pyplot as plt
import itertools
import pylab

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def plot_confusion_matrix(cm, classes, normalize=False, title='SAMM', cmap=plt.cm.Blues, font_size=12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Arguments:
    - cm : Confusion matrix values
    - classes : Class labels (list of strings)
    - normalize : If True, display percentages, else display counts
    - title : Title of the plot
    - cmap : Color map to use for the plot
    - font_size : Font size for annotations and labels
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    # Create heatmap of the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font_size + 2)

    # Add ticks for class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)

    # Format for displaying values, handle float formatting
    fmt = '.2f' if normalize else '.2f'  # Use .2f to display two decimals
    thresh = cm.max() / 2.

    # Annotate each cell in the confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=font_size)

    plt.tight_layout()
    #plt.ylabel('True label', fontsize=font_size)
    #plt.xlabel('Predicted label', fontsize=font_size)

    # Show the plot
    pylab.show()
"""
confusion_matrix = np.array([
    [0.74, 0.13, 0.13],
    [0.04, 0.84, 0.12],
    [0.16, 0.03, 0.81],
])

"""
# 使用提供的混淆矩阵值
confusion_matrix = np.array([
    [0.95, 0.03, 0.0, 0.02, 0.0],
    [0.12, 0.65, 0.08, 0.12, 0.03],
    [0.0, 0.33, 0.67, 0.0, 0.0],
    [0.08, 0.0, 0.04, 0.85, 0.03],
    [0.0, 0.07, 0.07, 0.13, 0.73]
])

# 类别标签
#labels = ['Disgust', 'Happiness', 'Repression', 'Surprise', 'Others']#casme ii
labels = ['Anger', 'Happiness', 'Contempt', 'Surprise', 'Others']#SAMM

#labels = ['Negative', 'Positive',  'Surprise']
# 绘制混淆矩阵
plot_confusion_matrix(confusion_matrix, classes=labels, normalize=False)
