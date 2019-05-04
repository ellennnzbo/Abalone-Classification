import numpy as np
import csv
import matplotlib.pyplot as plt

from data import import_for_classification
from fomlads.plot.evaluations import plot_roc
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import train_and_test_partition
from fomlads.model.classification import shared_covariance_model_fit
from fomlads.model.classification import shared_covariance_model_predict
from fomlads.model.classification import logistic_regression_fit
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

    N = inputs.shape[0]
    test_fraction = 0.2
    train_filter, test_filter = train_and_test_filter(N, test_fraction)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets, train_filter,
                                                                                      test_filter)
    # without basis functions
    print("WITHOUT BASIS FUNCTIONS")
    fig, ax = fit_and_plot_roc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_roc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig, ax), colour='b', type='training')
    fit_and_plot_roc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig, ax), colour='y', type='training')
    ax.legend(["Logistic regression", "Shared covariance model", "Fisher's linear discriminant"])
    fig.savefig('train_no_bf_roc')

    fig1, ax1 = fit_and_plot_roc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour='r', type='testing')
    fit_and_plot_roc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig1, ax1), colour='b', type='testing')
    fit_and_plot_roc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=(fig1, ax1), colour='y', type='testing')
    ax1.legend(["Logistic regression", "Shared covariance model", "Fisher's linear discriminant"])
    fig1.savefig('test_no_bf_roc')

    # with quadratic basis function
    print("WITH QUADRATIC BASIS FUNCTIONS")
    train_designmtx = quadratic_feature_mapping(train_inputs)
    test_designmtx = quadratic_feature_mapping(test_inputs)
    # train_designmtx = np.delete(train_designmtx, np.where(~train_designmtx.any(axis=0))[0], axis=1)
    # test_designmtx = np.delete(test_designmtx, np.where(~test_designmtx.any(axis=0))[0], axis=1)

    fig2, ax2 = fit_and_plot_roc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='training')
    fit_and_plot_roc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='y', type='training')
    ax2.legend(["Shared covariance model", "Fisher's linear discriminant"])
    fig2.savefig('train_quadratic_roc')

    fig3, ax3 = fit_and_plot_roc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='testing')
    fit_and_plot_roc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig3, ax3), colour='y', type='testing')
    ax3.legend(["Shared covariance model", "Fisher's linear discriminant"])
    fig3.savefig('test_quadratic_roc')

    plt.show()


def fit_and_plot_roc_logistic(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    weights = logistic_regression_fit(train_inputs, train_targets)
    thresholds = np.linspace(0,1,101)
    false_positive_rates = np.empty(thresholds.size)
    true_positive_rates = np.empty(thresholds.size)
    for i, threshold in enumerate(thresholds):
        if type == 'training':
            prediction_probs = logistic_regression_prediction_probs(train_inputs, weights)
            targets = train_targets
        elif type == 'testing':
            prediction_probs = logistic_regression_prediction_probs(test_inputs, weights)
            targets = test_targets
        predicts = (prediction_probs > threshold).astype(int)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        num_neg = np.sum(1 - targets)
        num_pos = np.sum(targets)
        false_positive_rates[i] = np.sum(num_false_positives)/num_neg
        true_positive_rates[i] = np.sum(num_true_positives)/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
    print("LOGISTIC REGRESSION ON", type, " DATA")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax


def fit_and_plot_roc_generative(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
    pi, mean0, mean1, covmtx = shared_covariance_model_fit(train_inputs, train_targets)
    thresholds = np.linspace(0,1,101)
    false_positive_rates = np.empty(thresholds.size)
    true_positive_rates = np.empty(thresholds.size)
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
        num_neg = np.sum(1 - targets)
        num_pos = np.sum(targets)
        false_positive_rates[i] = np.sum(num_false_positives)/num_neg
        true_positive_rates[i] = np.sum(num_true_positives)/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    # # and for the class prior we learnt from the model
    # predicts = shared_covariance_model_predict(
    #       inputs, pi, mean0, mean1, covmtx)
    # fpr = np.sum((predicts == 1) & (targets == 0))/num_neg
    # tpr = np.sum((predicts == 1) & (targets == 1))/num_pos
    # ax.plot([fpr], [tpr], 'rx', markersize=8, markeredgewidth=2)
    auc = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
    print("SHARED COVARIANCE MODEL ON", type, " DATA")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax

def fit_and_plot_roc_fisher(train_inputs, train_targets, test_inputs, test_targets, fig_ax=None, colour=None, type=None):
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
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    for i, w0 in enumerate(projected_inputs):
        false_positive_rates[i] = np.sum(1-targets[i:])/num_neg
        true_positive_rates[i] = np.sum(targets[i:])/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
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

