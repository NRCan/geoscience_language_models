# Copyright (C) 2021 ServiceNow, Inc.
""" Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, y):
    """Plot a confusion matrix
    
    :param cm: confusion matrix
    :param y: y_true, categorical series
    
    :returns: the image object, which will also be displayed
    """
    labels = list(y.cat.categories)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, np.round(cm[i, j],2),
                           ha="center", va="center", color="w")

    im = ax.imshow(cm)
    return im