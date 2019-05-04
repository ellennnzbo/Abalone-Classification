import numpy as np
import matplotlib.pyplot as plt

from data import import_for_classification
from fomlads.model.classification import fisher_linear_discriminant_projection
from fomlads.model.classification import project_data
from fomlads.plot.exploratory import plot_class_histograms

def main(ifname, input_cols=None, target_col=None, classes=None):
    """
    Import data and set aside test data
    """

    # import data
    inputs, targets, field_names, classes = import_for_classification(
        ifname, input_cols=input_cols, target_col=target_col, classes=classes)
    # plot fisher's projection
    weights = fisher_linear_discriminant_projection(inputs, targets)
    projected_data = project_data(inputs, weights)
    plot_class_histograms(projected_data, targets)

    plt.show()

    print(np.mean(targets))

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