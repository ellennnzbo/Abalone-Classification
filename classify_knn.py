import numpy as np
import csv
import matplotlib.pyplot as plt

from data import import_for_classification
from fomlads.evaluate.partition import train_and_test_partition
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import create_cv_folds
from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.plot.evaluations import plot_train_test_errors
from fomlads.evaluate.eval_classification import expected_loss
from sklearn.neighbors import KNeighborsClassifier

def main(ifname, input_cols=None, target_col=None, classes=None):
    """
    Import data and set aside test data
    """

    # import data
    inputs, targets, field_names, classes = import_for_classification(
        ifname, input_cols=input_cols, target_col=target_col, classes=classes)

    # split into training and test data
    N = inputs.shape[0]
    test_fraction = 0.2
    train_filter, test_filter = train_and_test_filter(N, test_fraction)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter, test_filter)

    num_folds=5

    # knn model
    n_neighbours_sequence = np.array(np.arange(1, 50))
    best_n = evaluate_n_neighbours(inputs, targets, num_folds, n_neighbours_sequence=n_neighbours_sequence)


def evaluate_n_neighbours(inputs, targets, num_folds, n_neighbours_sequence=None):
    lossmtx = np.matrix([[0, 1000], [1, 0]])
    N = inputs.shape[0]
    folds = create_cv_folds(N, num_folds)
    if n_neighbours_sequence is None:
        n_neighbours_sequence = np.arange(1, 25)
    num_values = n_neighbours_sequence.size
    test_errors = np.zeros(num_values)
    train_errors = np.zeros(num_values)
    test_errors_stes = np.zeros(num_values)
    train_errors_stes = np.zeros(num_values)
    test_losses = np.zeros(num_values)
    train_losses = np.zeros(num_values)
    test_losses_stes = np.zeros(num_values)
    train_losses_stes = np.zeros(num_values)
    # cross validation
    for n, n_neighbours in enumerate(n_neighbours_sequence):
        knn = KNeighborsClassifier(n_neighbors=n_neighbours)
        test_error_per_fold = np.zeros(num_folds)
        train_error_per_fold = np.zeros(num_folds)
        test_loss_per_fold = np.zeros(num_folds)
        train_loss_per_fold = np.zeros(num_folds)
        for f, fold in enumerate(folds):
            train_part, test_part = fold
            train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_part, test_part)
            knn.fit(train_inputs, train_targets)
            # compute training errors and loss
            train_predicts = knn.predict(train_inputs)
            train_error_per_fold[f] = misclassification_error(train_targets, train_predicts)
            train_loss_per_fold[f] = expected_loss(train_targets, train_predicts, lossmtx)
            # compute testing errors and loss
            test_predicts = knn.predict(test_inputs)
            test_error_per_fold[f] = misclassification_error(test_targets, test_predicts)
            test_loss_per_fold[f] = expected_loss(test_targets, test_predicts, lossmtx)
        # mean errors
        train_errors[n] = np.mean(train_error_per_fold)
        test_errors[n] = np.mean(test_error_per_fold)
        test_errors_stes[n] = np.std(test_error_per_fold)/np.sqrt(num_folds)
        train_errors_stes[n] = np.std(train_error_per_fold) / np.sqrt(num_folds)
        # mean losses
        train_losses[n] = np.mean(train_loss_per_fold)
        test_losses[n] = np.mean(test_loss_per_fold)
        test_losses_stes[n] = np.std(test_loss_per_fold) / np.sqrt(num_folds)
        train_losses_stes[n] = np.std(train_loss_per_fold) / np.sqrt(num_folds)
    fig, ax = plot_train_test_errors(
        "Neighbours", n_neighbours_sequence, train_errors, test_errors,
        train_stes=train_errors_stes, test_stes=test_errors_stes)
    fig.savefig('knn2.png')
    min_error_neighbour = n_neighbours_sequence[np.argmin(test_errors)]
    print("N NEIGHBOURS WITH MINIMUM MISCLASSIFICATION ERROR: ", min_error_neighbour)
    plt.show()
    return min_error_neighbour

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments
    else:
        # assumes that the first argument is the input filename/path
        if len(sys.argv) == 2:
            main(ifname=sys.argv[1])
        else:
            # assumes that the second argument is a comma separated list of
            # the classes to plot
            classes = sys.argv[2].split(',')
            if len(sys.argv) == 3:
                main(ifname=sys.argv[1], classes=classes)
            else:
                # assumes that the third argument is the target column
                target_col = sys.argv[3]
                if len(sys.argv) == 4:
                    main(
                        ifname=sys.argv[1], classes=classes,
                        target_col=target_col)
                # assumes that the fourth argument is the list of input columns
                else:
                    input_cols = sys.argv[4].split(',')
                    main(
                        ifname=sys.argv[1], classes=classes,
                        input_cols=input_cols, target_col=target_col)

