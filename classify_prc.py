import numpy as np
import csv
import matplotlib.pyplot as plt

from data import import_for_classification
from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms
from fomlads.plot.evaluations import plot_roc
from fomlads.plot.evaluations import plot_prc

from fomlads.evaluate.partition import create_cv_folds
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import train_and_test_partition

from fomlads.model.classification import shared_covariance_model_fit
from fomlads.model.classification import shared_covariance_model_predict
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs
from fomlads.model.classification import project_data
from fomlads.model.classification import fisher_linear_discriminant_projection
from fomlads.model.basis_functions import quadratic_feature_mapping


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
    # print("input = %r " % (inputs))
    # print("target = %r " % (targets))
    # print("field names = %r" % (field_names))
    # print("classes = %r " % (classes))
    # print('N x D: ', inputs.shape)

    N = inputs.shape[0]
    test_fraction = 0.2
    train_filter, test_filter = train_and_test_filter(N, test_fraction)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                      test_filter)

    fig, ax = fit_and_plot_prc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_prc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig, ax), colour='b', type='training')
    fit_and_plot_prc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig, ax), colour='y', type='training')
    ax.legend(["Logistic regression", "Shared covariance model", "Fisher's linear discriminant"])
    fig.savefig('train_no_bf_prc')

    fig1, ax1 = fit_and_plot_prc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='testing')
    fit_and_plot_prc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig1, ax1), colour='b', type='testing')
    fit_and_plot_prc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig1, ax1), colour='y', type='testing')
    ax1.legend(["Logistic regression", "Shared covariance model", "Fisher's linear discriminant"])
    fig1.savefig('test_no_bf_prc')


    # with quadratic basis function
    train_designmtx = quadratic_feature_mapping(train_inputs)
    test_designmtx = quadratic_feature_mapping(test_inputs)
    # train_designmtx = np.delete(train_designmtx, np.where(~train_designmtx.any(axis=0))[0], axis=1)
    # test_designmtx = np.delete(test_designmtx, np.where(~test_designmtx.any(axis=0))[0], axis=1)

    fig2, ax2 = fit_and_plot_prc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='training')
    fit_and_plot_prc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='y', type='training')
    # fit_and_plot_prc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='r', type='training')
    ax2.legend(["Shared covariance model", "Fisher's linear discriminant"])
    fig2.savefig('train_quadratic_prc')


    fig3, ax3 = fit_and_plot_prc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='testing')
    fit_and_plot_prc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig3, ax3), colour='y', type='testing')
    # fit_and_plot_prc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig3, ax3), colour='r', type='testing')
    ax3.legend(["Shared covariance model", "Fisher's linear discriminant"])
    fig3.savefig('test_quadratic_roc')


    plt.show()

def fit_and_plot_prc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    weights = logistic_regression_fit(train_inputs, train_targets)
    thresholds = np.linspace(0, 1, 101)
    precision_values = np.zeros(len(thresholds))
    recall_values = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        if type == 'training':
            inputs = train_inputs
            targets = train_targets
        elif type == 'testing':
            inputs = test_inputs
            targets = test_targets
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > threshold).astype(int)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        num_false_negatives = np.sum((predicts == 0) & (targets == 1))
        precision_values[i] = num_true_positives/(num_true_positives+num_false_positives)
        recall_values[i] = num_true_positives/(num_true_positives+num_false_negatives)
    fig, ax = plot_prc(
            recall_values, precision_values, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(precision_values, recall_values)
    print("LOGISTIC REGRESSION ON", type, " DATA")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax


def fit_and_plot_prc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    pi, mean0, mean1, covmtx = shared_covariance_model_fit(train_inputs, train_targets)
    thresholds = np.linspace(0, 1, 101)
    precision_values = np.zeros(len(thresholds))
    recall_values = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        if type == 'training':
            inputs = train_inputs
            targets = train_targets
        elif type == 'testing':
            inputs = test_inputs
            targets = test_targets
        predicts = shared_covariance_model_predict(
            inputs, pi, mean0, mean1, covmtx, threshold)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        num_false_negatives = np.sum((predicts == 0) & (targets == 1))
        precision_values[i] = num_true_positives / (num_true_positives + num_false_positives)
        recall_values[i] = num_true_positives / (num_true_positives + num_false_negatives)
    fig, ax = plot_prc(
        recall_values, precision_values, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(np.flip(precision_values), np.flip(recall_values))
    print("SHARED COVARIANCE MODEL ON", type, " DATA")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax

def fit_and_plot_prc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    weights = fisher_linear_discriminant_projection(train_inputs, train_targets)
    if type == 'training':
        inputs = train_inputs
        targets = train_targets
    elif type == 'testing':
        inputs = test_inputs
        targets = test_targets
    projected_inputs = project_data(inputs, weights)
    # sort project_inputs in ascending order and sort targets accordingly
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    targets = np.copy(targets[new_ordering])
    N = targets.size
    precision_values = np.empty(N)
    recall_values = np.empty(N)
    for i, w0 in enumerate(projected_inputs):
        num_false_positives = np.sum(1-targets[i:])
        num_true_positives = np.sum(targets[i:])
        num_false_negatives = np.sum(targets[:i])
        precision_values[i] = num_true_positives / (num_true_positives + num_false_positives)
        recall_values[i] = num_true_positives / (num_true_positives + num_false_negatives)
    fig, ax = plot_prc(
        recall_values, precision_values, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(np.flip(precision_values), np.flip(recall_values))
    print("FISHER'S LINEAR DISCRIMINANT ON", type, " DATA")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
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