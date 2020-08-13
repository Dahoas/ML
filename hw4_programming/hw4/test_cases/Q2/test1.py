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

    lamb = 0.1
    w0 = np.array([[7.0], [1.5]])
    num_iter = 10

    alpha_unit_test_values = [3, 0.9, 0.3]
    colors = ['steelblue', 'green', 'orange']

    # Solution values
    expected_w_lists = {}
    expected_w_lists[3] = [ (7.000,1.500), (1.901,-4.303), (4.081,14.699), (-0.143,4.289), (-3.053,-2.910),
            (0.855,15.799), (-2.402,5.059), (-4.484,-2.243), (-0.143,16.128), (-3.100,5.289), (-4.868,-1.980) ]
    expected_w_lists[0.9] = [ (7.000,1.500), (5.470,-0.241), (4.090,0.330), (2.839,-0.365), (1.887,0.894),
            (0.879,-0.472), (0.828,2.561), (-0.117,0.627), (-0.537,0.175), (-0.405,1.664), (-1.060,0.102) ]
    expected_w_lists[0.3] = [ (7.000,1.500), (6.490,0.920), (5.996,0.431), (5.516,0.131), (5.053,0.034), 
            (4.605,0.014), (4.172,0.015), (3.756,0.021), (3.357,0.031), (2.975,0.044), (2.612,0.062) ]

    util.plot_objective_contours(X, y, lamb, title='Gradient Descent', colors='gray',
            show_labels=False, new_figure=True, show_figure=False, save_filename=None)

    for alpha, color in zip(alpha_unit_test_values, colors):
        actual_w_list = logistic_regression.gradient_descent(X, y, lamb, alpha, w0, num_iter)

        expected_w_list = expected_w_lists[alpha]

        for i in range(num_iter+1):
            assert abs(actual_w_list[i][0,0] - expected_w_list[i][0]) < 0.01 , 'Incorrect weight value found for iter={}, w[0]. Expected w={}, found w={}'.format(i, expected_w_list[i], actual_w_list[i])
            assert abs(actual_w_list[i][1,0] - expected_w_list[i][1]) < 0.01 , 'Incorrect weight value found for iter={}, w[1]. Expected w={}, found w={}'.format(i, expected_w_list[i], actual_w_list[i])
 

        util.plot_optimization_path(actual_w_list, color=color, label='alpha = {:.1f}'.format(alpha))

    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend(fontsize=util.get_font_size())

    filename = '{}/gradient_descent.png'.format(figures_directory)
    plt.savefig(filename)

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

if __name__ == "__main__":
    test()
