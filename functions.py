import numpy as np
def remove_outliers(X, labels, y):
    outliers1 = np.where(labels == -1)[0]
    # Remove the outliers from the data.
    X1 = np.delete(X, outliers1, axis=0)
    # Remove the outliers from the predicted labels.

    y1 = np.delete(labels, outliers1, axis=0)

    # Remove the outliers from the ground truth labels.
    y_true1 = np.delete(y, outliers1, axis=0)

    return X1, y1, y_true1

