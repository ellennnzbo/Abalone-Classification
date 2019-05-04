import numpy as np
import csv
import matplotlib.pyplot as plt

from data import import_for_classification
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import train_and_test_partition
from fomlads.evaluate.eval_classification import misclassification_error

from fomlads.model.classification import shared_covariance_model_fit
from fomlads.model.classification import shared_covariance_model_predict
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_prediction_probs
from fomlads.model.basis_functions import quadratic_feature_mapping
from fomlads.plot.evaluations import plot_misclassification_errors

def main(
        ifname, input_cols=None, target_col=None, classes=None):
    """
    Imports the titanic data-set and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    """
    inputs, targets, field_names, classes = import_for_classification(
        ifname, input_cols=input_cols, target_col=target_col, classes=classes)

    N = inputs.shape[0]
    test_fraction = 0.2
    train_filter, test_filter = train_and_test_filter(N, test_fraction)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                      test_filter)
    # # without basis functions
    fig0, ax0 = fit_and_plot_accuracy_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_accuracy_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig0, ax0), colour='b',  type='testing')
    ax0.legend(["Training", "Testing"])
    fig0.savefig('logistic_no_bf_accuracy.png')

    fig1, ax1 = fit_and_plot_accuracy_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_accuracy_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig1, ax1), colour='b', type='testing')
    ax1.legend(["Training", "Testing"])
    fig1.savefig('generative_no_bf_accuracy.png')

    # with quadratic basis function
    train_designmtx = quadratic_feature_mapping(train_inputs)
    test_designmtx = quadratic_feature_mapping(test_inputs)
    # train_designmtx = np.delete(train_designmtx, np.where(~train_designmtx.any(axis=0))[0], axis=1)
    # test_designmtx = np.delete(test_designmtx, np.where(~test_designmtx.any(axis=0))[0], axis=1)

    print("WITH QUADRATIC BASIS FUNCTIONS")
    fig2, ax2 = fit_and_plot_accuracy_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_accuracy_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='b', type='testing')
    ax2.legend(["Training", "Testing"])
    fig2.savefig('generative_quadratic_accuracy.png')

    # # fig, ax0, ax1= fit_and_plot_accuracy_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='r', type='training')
    # fit_and_plot_accuracy_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig, ax0, ax1), colour='b', type='testing')

    plt.show()


def fit_and_plot_accuracy_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    weights = logistic_regression_fit(train_inputs, train_targets)
    if type == 'training':
        inputs = train_inputs
        targets = train_targets
    elif type == 'testing':
        inputs = test_inputs
        targets = test_targets
    thresholds = np.linspace(0, 1, 101)
    errors = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > threshold).astype(int)
        errors[i] = misclassification_error(targets, predicts)
    fig, ax = plot_misclassification_errors("Thresholds", thresholds, errors, fig_ax=fig_ax, colour=colour)
    min_index = np.argmin(errors)
    min_threshold = thresholds[min_index]
    print("LOGISTIC REGRESSION")
    print("THRESHOLD WITH MINIMUM ERROR IN ", type, " DATA: ", min_threshold)
    print("ERROR AT THRESHOLD: ", min(errors))
    print("")
    return fig, ax


def fit_and_plot_accuracy_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    pi, mean0, mean1, covmtx = shared_covariance_model_fit(train_inputs, train_targets)
    if type == 'training':
        inputs = train_inputs
        targets = train_targets
    elif type == 'testing':
        inputs = test_inputs
        targets = test_targets
    thresholds = np.linspace(0, 1, 101)
    errors = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        predicts = shared_covariance_model_predict(
            inputs, pi, mean0, mean1, covmtx, threshold)
        errors[i] = misclassification_error(targets, predicts)
    fig, ax = plot_misclassification_errors("Thresholds", thresholds, errors, fig_ax=fig_ax, colour=colour)
    min_index = np.argmin(errors)
    min_threshold = thresholds[min_index]
    print("SHARED COVARIANCE MODEL")
    print("THRESHOLD WITH MINIMUM ERROR IN ", type, " DATA: ", min_threshold)
    print("ERROR AT THRESHOLD: ", min(errors))
    print("")
    return fig, ax


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