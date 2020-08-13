import numpy as np
import matplotlib.pyplot as plt
import os

import util

import logistic_regression

def max_score():
    return 2

def timeout():
    return 60

def test():

    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    X, y = logistic_regression.setup_data()

    lambda_unit_test_values = [0, 0.1, 1, 10]
    w_unit_test_values = [ np.array([[0.0], [0.0]]), np.array([[2.0], [2.0]]), np.array([[-2.0], [4.0]]) ]

    # Solution values
    expected_output_values = {}
    expected_output_values[0] = [2.773, 6.148, 6.145]
    expected_output_values[0.1] = [2.773, 6.548, 7.145]
    expected_output_values[1] = [2.773, 10.148, 16.145]
    expected_output_values[10] = [2.773, 46.148, 106.145]

    for lamb in lambda_unit_test_values:
        for i, w in enumerate(w_unit_test_values):
            actual_output = logistic_regression.objective(w, X, y, lamb)
            if isinstance(actual_output, np.ndarray):
                actual_output = actual_output[0, 0]

            expected_output = expected_output_values[lamb][i]

            assert abs(actual_output - expected_output) < 0.01 , 'Incorrect objective value found for lamda={}, w={}. Expected {}, found {}'.format(lamb, w, expected_output, actual_output)
 

        filename = '{}/objective_lambda_{:0.1f}.png'.format(figures_directory, lamb)
        filename = filename.replace('.', '_', 1)
        title = 'lambda = {}'.format(lamb)
        util.plot_objective_contours(X, y, lamb, title=title,
                new_figure=True, show_figure=False, save_filename=filename)

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

if __name__ == "__main__":
    test()
