import numpy as np
import matplotlib.pyplot as plt


from data import import_for_classification
from fomlads.evaluate.partition import train_and_test_filter
from fomlads.evaluate.partition import train_and_test_partition
from fomlads.evaluate.partition import create_cv_folds
from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.model.classification import fisher_linear_discriminant_projection
from fomlads.model.classification import project_data
from fomlads.model.clustering import kmeans
from fomlads.model.basis_functions import construct_rbf_feature_mapping
from fomlads.model.basis_functions import quadratic_feature_mapping
from fomlads.plot.evaluations import plot_roc
from classify_roc import fit_and_plot_roc_fisher
from classify_roc import fit_and_plot_roc_logistic
from classify_roc import fit_and_plot_roc_generative
from classify_prc import fit_and_plot_prc_fisher
from classify_prc import fit_and_plot_prc_logistic
from classify_prc import fit_and_plot_prc_generative
from classify_accuracy import fit_and_plot_accuracy_generative
from classify_accuracy import fit_and_plot_accuracy_logistic
from classify_knn import evaluate_n_neighbours
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

    m_sequence = np.array(np.arange(10,20))
    s_sequence = np.array(np.linspace(0, 1, 50))
    num_folds = 5
    fig0, ax0 = fit_and_plot_roc(train_inputs, train_targets, fig_ax=None, colour='r')
    fit_and_plot_quadraticbf_roc(train_inputs, train_targets, fig_ax=(fig0, ax0), colour='y')
    fit_and_plot_rbf_roc(train_inputs, train_targets, fig_ax=(fig0, ax0), colour='m', m_sequence=m_sequence, s_sequence=s_sequence, num_folds=num_folds)
    ax0.legend(["No basis functions", "Quadratic basis function", "Radial basis functions"])

    # evaluate model performance with basis function parameters
    print("EVALUATE ALL MODELS WITH RBF")
    m_best, s_best = fit_rbf(train_inputs, train_targets, m_sequence=m_sequence, s_sequence=s_sequence, num_folds=num_folds)
    centres, _ = kmeans(inputs, int(m_best), initial_centres=None, threshold=0.01, iterations=None)
    centres = np.transpose(centres)
    feature_mapping = construct_rbf_feature_mapping(centres, s_best)
    train_designmtx = feature_mapping(train_inputs)
    test_designmtx = feature_mapping(test_inputs)

    # with ROC
    fig1, ax1 = fit_and_plot_roc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='training')
    fit_and_plot_roc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig1, ax1), colour='y', type='training')
    fit_and_plot_roc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig1, ax1), colour='r', type='training')
    ax1.legend(["Shared covariance model", "Fisher's linear discriminant", "Logistic Regression"])

    fig2, ax2 = fit_and_plot_roc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='testing')
    fit_and_plot_roc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='y', type='testing')
    fit_and_plot_roc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig2, ax2), colour='r', type='testing')
    ax2.legend(["Shared covariance model", "Fisher's linear discriminant", "Logistic Regression"])

    # with PRC
    fig3, ax3 = fit_and_plot_prc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='training')
    fit_and_plot_prc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig3, ax3), colour='y', type='training')
    fit_and_plot_prc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig3, ax3), colour='r', type='training')
    ax3.legend(["Shared covariance model", "Fisher's linear discriminant", "Logistic Regression"])

    fig4, ax4 = fit_and_plot_prc_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='b', type='testing')
    fit_and_plot_prc_fisher(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig4, ax4), colour='y', type='testing')
    fit_and_plot_prc_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig4, ax4), colour='r', type='testing')
    ax4.legend(["Shared covariance model", "Fisher's linear discriminant", "Logistic Regression"])

    # with misclassification error
    fig5, ax5 = fit_and_plot_accuracy_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_accuracy_generative(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig5, ax5), colour='b', type='testing')
    ax5.legend(["Training", "Testing"])

    fig6, ax6= fit_and_plot_accuracy_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=None, colour='r', type='training')
    fit_and_plot_accuracy_logistic(train_designmtx, train_targets, test_designmtx, test_targets, fig_ax=(fig6, ax6), colour='b', type='testing')
    ax6.legend(["Training", "Testing"])

    # fit knn with best n, m and s
    n_neighbours_sequence = np.array(np.arange(1, 50))
    best_n = evaluate_n_neighbours(train_designmtx, train_targets, num_folds, n_neighbours_sequence=n_neighbours_sequence)
    knn = KNeighborsClassifier(n_neighbors=best_n)
    knn.fit(train_designmtx, train_targets)
    predicts = knn.predict(test_designmtx)
    error = misclassification_error(test_targets, predicts)
    print(error)

    plt.show()

def fit_and_plot_roc(inputs, targets, fig_ax=None, colour=None):
    weights = fisher_linear_discriminant_projection(inputs, targets)
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
    print("NO BASIS FUNCTIONS")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax

def fit_and_plot_quadraticbf_roc(inputs, targets, fig_ax=None, colour=None):
    # quadratic feature mapping
    designmtx = quadratic_feature_mapping(inputs)
    # fisher's discriminant
    N = designmtx.shape[0]
    weights = fisher_linear_discriminant_projection(designmtx, targets)
    projected_inputs = project_data(designmtx, weights)
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    targets = np.copy(targets[new_ordering])
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    num_neg = np.sum(1 - targets)
    num_pos = np.sum(targets)
    for i, w0 in enumerate(projected_inputs):
        false_positive_rates[i] = np.sum(1 - targets[i:]) / num_neg
        true_positive_rates[i] = np.sum(targets[i:]) / num_pos
    fig, ax = plot_roc(false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    auc = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
    print("QUADRATIC BASIS FUNCTION")
    print("AREA UNDER CURVE: ", auc)
    print(" ")
    return fig, ax

def fit_and_plot_rbf_roc(inputs, targets, fig_ax=None, colour=None, m_sequence=None, s_sequence=None, num_folds=None):
    """
    Takes sequences of M and S
    Creates feature mapping for each combination
    Uses design mtx for fisher's linear discriminant
    Calculates misclassification error using cross-validation

    Returns:
        average misclassification error and standard error
    """
    if m_sequence is None:
        m_sequence = np.arange(3, 20)
    if s_sequence is None:
        s_sequence = np.logspace(-2, 0)

    # initialise grid of parameters
    grid = np.array(np.meshgrid(m_sequence, s_sequence)).T.reshape(-1, 2)
    n_grid = grid.shape[0]

    # grid search with cv
    auc = np.zeros(n_grid)
    std_errors = np.zeros(n_grid)
    for i, m_s_pair in enumerate(grid):
        auc_per_parameter = np.zeros(num_folds)
        # create design matrix
        m = int(m_s_pair[0])
        s = m_s_pair[1]
        centres, _ = kmeans(inputs, m, initial_centres=None, threshold=0.01, iterations=None)
        centres = np.transpose(centres)
        feature_mapping = construct_rbf_feature_mapping(centres, s)
        designmtx = feature_mapping(inputs)

        # cross validation
        N = designmtx.shape[0]
        folds = create_cv_folds(N, num_folds)
        for f, fold in enumerate(folds):
            train_part, test_part = fold
            train_designmtx, train_targets, test_designmtx, test_targets = train_and_test_partition(designmtx, targets,
                                                                                                    train_part,
                                                                                                    test_part)
            weights = fisher_linear_discriminant_projection(train_designmtx, train_targets)
            projected_inputs = project_data(test_designmtx, weights)
            new_ordering = np.argsort(projected_inputs)
            projected_inputs = projected_inputs[new_ordering]
            test_targets = np.copy(test_targets[new_ordering])
            num_neg = np.sum(1 - test_targets)
            num_pos = np.sum(test_targets)
            test_n = test_targets.shape[0]
            false_positive_rates = np.empty(test_n)
            true_positive_rates = np.empty(test_n)
            for w, w0 in enumerate(projected_inputs):
                false_positive_rates[w] = np.sum(1 - test_targets[w:]) / num_neg
                true_positive_rates[w] = np.sum(test_targets[w:]) / num_pos
            auc_per_parameter[f] = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
        fig, ax = plot_roc(false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour, linewidth=0.3)
        auc[i] = np.mean(auc_per_parameter)
        std_errors[i] = np.std(auc_per_parameter) / np.sqrt(num_folds)
    max_auc = max(auc)
    best_m_s_pair = grid[np.argmax(auc)]
    best_m = best_m_s_pair[0]
    best_s = best_m_s_pair[1]
    print("RADIAL BASIS FUNCTIONS CROSS VALIDATION RESULTS")
    print("MAX AREA UNDER CURVE: ", max_auc)
    print("number of centres: ", best_m, ", scale: ", best_s)
    return fig, ax

def fit_rbf(inputs, targets, m_sequence=None, s_sequence=None, num_folds=None):
    """
    Takes sequences of M and S
    Creates feature mapping for each combination
    Uses design mtx for fisher's linear discriminant
    Calculates misclassification error using cross-validation

    Returns:
        average misclassification error and standard error
    """
    if m_sequence is None:
        m_sequence = np.arange(3, 20)
    if s_sequence is None:
        s_sequence = np.logspace(-2, 0)

    # initialise grid of parameters
    grid = np.array(np.meshgrid(m_sequence, s_sequence)).T.reshape(-1, 2)
    n_grid = grid.shape[0]

    # grid search with cv
    auc = np.zeros(n_grid)
    std_errors = np.zeros(n_grid)
    for i, m_s_pair in enumerate(grid):
        auc_per_parameter = np.zeros(num_folds)
        # create design matrix
        m = int(m_s_pair[0])
        s = m_s_pair[1]
        centres, _ = kmeans(inputs, m, initial_centres=None, threshold=0.01, iterations=None)
        centres = np.transpose(centres)
        feature_mapping = construct_rbf_feature_mapping(centres, s)
        designmtx = feature_mapping(inputs)

        # cross validation
        N = designmtx.shape[0]
        folds = create_cv_folds(N, num_folds)
        for f, fold in enumerate(folds):
            train_part, test_part = fold
            train_designmtx, train_targets, test_designmtx, test_targets = train_and_test_partition(designmtx, targets,
                                                                                                    train_part,
                                                                                                    test_part)
            weights = fisher_linear_discriminant_projection(train_designmtx, train_targets)
            projected_inputs = project_data(test_designmtx, weights)
            new_ordering = np.argsort(projected_inputs)
            projected_inputs = projected_inputs[new_ordering]
            test_targets = np.copy(test_targets[new_ordering])
            num_neg = np.sum(1 - test_targets)
            num_pos = np.sum(test_targets)
            test_n = test_targets.shape[0]
            false_positive_rates = np.empty(test_n)
            true_positive_rates = np.empty(test_n)
            for w, w0 in enumerate(projected_inputs):
                false_positive_rates[w] = np.sum(1 - test_targets[w:]) / num_neg
                true_positive_rates[w] = np.sum(test_targets[w:]) / num_pos
            auc_per_parameter[f] = np.trapz(np.flip(true_positive_rates), np.flip(false_positive_rates))
        auc[i] = np.mean(auc_per_parameter)
        std_errors[i] = np.std(auc_per_parameter) / np.sqrt(num_folds)
    max_auc = max(auc)
    best_m_s_pair = grid[np.argmax(auc)]
    best_m = best_m_s_pair[0]
    best_s = best_m_s_pair[1]
    print("RADIAL BASIS FUNCTIONS CROSS VALIDATION RESULTS")
    print("MAX AREA UNDER CURVE: ", max_auc)
    print("number of centres: ", best_m, ", scale: ", best_s)
    return best_m, best_s

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