from __future__ import print_function
import numpy as np
from random import randrange

EPSILON_UPDATE_FACTOR = 0.5


def epsilon_update(epsilon):
    return EPSILON_UPDATE_FACTOR*epsilon


def gradient_test(f, f_gradient, x, epsilon0, num_iter=30, delta=0.1):
    d = np.random.randn(*x.shape) * (2 / np.sqrt(np.size(x)))
    total_err1 = 0
    total_err2 = 0
    flag = True
    epsilon = epsilon0
    grad = f_gradient(x) # grad by x requires trnapose here....
    prev_value1 = np.linalg.norm(f(x+epsilon*d)-f(x))
    prev_value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-epsilon*np.dot(grad, epsilon*d.T))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
        value2 = np.linalg.norm(f(x+epsilon*d)-f(x)-epsilon*np.dot(grad, epsilon*d.T))
        total_err1 = total_err1+value1
        total_err2 = total_err2+value2
        print (["Gradient test iteration ", i, "  :  ", value1/prev_value1, value2/prev_value2])
        if np.abs(value2/prev_value2 - EPSILON_UPDATE_FACTOR**2) > delta or np.abs(value1/prev_value1 - EPSILON_UPDATE_FACTOR) > delta:
            flag = False
        prev_value1 = value1
        prev_value2 = value2
    return total_err1,total_err2, flag


def jacobian_test(f, f_jacobianmv, x, epsilon0, num_iter=30, delta=0.1,dim_d=None):
    if dim_d==None:
        d = np.random.randn(*x.shape) * (2 / np.sqrt(np.size(x)))
    else:
        assert x.shape[1]==dim_d[0]
        d = np.random.randn(*dim_d) * (2 / np.sqrt(np.size(dim_d[0] * dim_d[1])))

    total_err1 = 0
    total_err2 = 0
    passed_all_tests = True
    epsilon = epsilon0
    if(dim_d==None):
        prev_value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
        prev_value2 = np.linalg.norm(f(x + epsilon * d) - f(x) - f_jacobianmv(x, epsilon * d))
    else:
        res=np.add(x,epsilon * d.T)
        ress1= f(np.add(x,epsilon * d.T))
        resss2=f(x)
        prev_value1 = np.linalg.norm(f(np.add(x,epsilon * d.T)) - f(x))
        prev_value2 = np.linalg.norm(f(np.add(x,epsilon * d.T)) - f(x) - f_jacobianmv(x, epsilon * d))
    for i in range(0, num_iter):
        epsilon = epsilon_update(epsilon)
        if (dim_d == None):
            value1 = np.linalg.norm(f(x + epsilon * d) - f(x))
            value2 = np.linalg.norm(f(x + epsilon * d) - f(x) - f_jacobianmv(x, epsilon * d))
        else:
            value1 = np.linalg.norm(f(np.add(x, epsilon * d.T)) - f(x))
            value2 = np.linalg.norm(f(np.add(x, epsilon * d.T)) - f(x) - f_jacobianmv(x, epsilon * d))
        total_err1 = total_err1+value1
        total_err2 = total_err2+value2
        conv_rate_1 = value1 / prev_value1
        conv_rate_2 = value2 / prev_value2
        print(["Jacobian test iteration ", i, "  :  ", conv_rate_1, conv_rate_2])
        if np.abs(conv_rate_2 - EPSILON_UPDATE_FACTOR**2) > delta or np.abs(conv_rate_1 - EPSILON_UPDATE_FACTOR) > delta:
            passed_all_tests = False

        prev_value1 = value1
        prev_value2 = value2
    return total_err1, total_err2, passed_all_tests


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """

    total_err = 0

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        total_err += rel_error
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

    return total_err

def grad_check_full(f, x, analytic_grad, num_checks=10, h=1e-3):
    """
    Experimental test...
    """

    total_err = 0

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x.copy()
        x = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x = oldval  # reset

        grad_numerical = (fxph - fxmh)
        grad_analytic = np.sum(2*h*analytic_grad)
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        total_err += rel_error
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

        h = h*0.5

    return total_err