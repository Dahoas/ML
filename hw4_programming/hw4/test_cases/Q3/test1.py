import numpy as np
import matplotlib.pyplot as plt
import os

import util

import logistic_regression

def max_score():
    return 4

def timeout():
    return 60

def test():

    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    X, y = logistic_regression.setup_data()

    lambda_values = [0.1, 2]
    alpha_values = [0.5, 0.2]

    w0 = np.array([[7.0], [1.5]])
    num_iter = 10

    colors = ['green', 'darkorchid']

    # Solution values
    expected_w_list = [ (7.000,1.500), (-0.499,-0.628), (-0.020,0.777), (-0.139,0.180), (-0.122,0.310), (-0.121,0.315), (-0.121,0.315), (-0.121,0.315), (-0.121,0.315), (-0.121,0.315), (-0.121,0.315) ]

    for lamb, alpha in zip(lambda_values, alpha_values):

        util.plot_objective_contours(X, y, lamb, title='Newton\'s Method vs. Gradient Descent, lambda={}'.format(lamb), colors='gray',
                show_labels=False, new_figure=True, show_figure=False, save_filename=None)

        gd_w_list = logistic_regression.gradient_descent(X, y, lamb, alpha, w0, num_iter)
        util.plot_optimization_path(gd_w_list, color=colors[0], label='Gradient Descent')

        actual_w_list = logistic_regression.newtons_method(X, y, lamb, w0, num_iter)


        util.plot_optimization_path(actual_w_list, color=colors[1], label='Newton\'s Method')

        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        plt.legend(fontsize=util.get_font_size())

        filename = '{}/newtons_method_lambda_{:.1f}.png'.format(figures_directory, lamb)
        filename = filename.replace('.', '_', 1)
        plt.savefig(filename)


    for i in range(num_iter+1):
        assert abs(actual_w_list[i][0,0] - expected_w_list[i][0]) < 0.01 , 'Incorrect weight value found for iter={}, w[0]. Expected w={}, found w={}'.format(i, expected_w_list[i], actual_w_list[i])
        assert abs(actual_w_list[i][1,0] - expected_w_list[i][1]) < 0.01 , 'Incorrect weight value found for iter={}, w[1]. Expected w={}, found w={}'.format(i, expected_w_list[i], actual_w_list[i])
 

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

if __name__ == "__main__":
    test()
