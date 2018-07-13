import numpy as np
import tables
import sys,os
import scipy.io
from softmax import Softmax, ResNetwork, train_with_sgd,ResLayer
import matplotlib.pyplot as plt
import itertools
from gradient_tests import grad_check_full, grad_check_sparse, gradient_test, jacobian_test

TEST_MODE = True


def run_tests():
    pass


def generate_all_combinations(grid):
    return itertools.product(*list(grid.values()))


def run_tests(sample, label, num_labels):

    ############# TODO REFACTOR!!!!!! ############
    dim = sample.shape[1]

    ###########  SOFTMAX TESTS  ############

    #demo_softmax = LonelySoftmaxWithReg(dim=dim, num_labels=num_labels, reg_param=0.1)
    demo_softmax= Softmax(dim=dim, num_labels=num_labels, reg_param=0.1)
    w = demo_softmax.get_params()

    ##1. Test softmax gradient w.r.t X ##

    softmax_val_x = lambda x: demo_softmax.calc_value_and_grad(x, label, calc_value=True, calc_grad_by_params=False)[0]
    softmax_grad_x = lambda x: demo_softmax.calc_grad_by_x(x, label).T # returns (dL\dX)
  #  gradient_test(softmax_val_x, softmax_grad_x, sample,epsilon0=50, num_iter=10, delta=0.1)

    #2. Softmax Jacobian test
    f_jacobianmv = lambda x,v: np.dot(softmax_grad_x(x),v.T)[0,0]
    #jacobian_test(softmax_val_x, f_jacobianmv, sample, epsilon0=50, num_iter=30, delta=0.1)


    #Test softmax gradient w.r.t X numerically
    #grad = demo_softmax.calc_grad_by_x(sample, label)
    #grad_err_1 = grad_check_full(softmax_val_x, sample, grad, 10)
    #grad_err_2 = grad_check_sparse(softmax_val_x, sample, grad, 10)
    #print(['grad_err_1:',grad_err_1,'grad_err_2',grad_err_2])
    #assert grad_err_2 < 0.01

    ##3. Test softmax gradient w.r.t Params ##

    softmax_val_w = lambda w: demo_softmax.calc_value_and_grad(sample, label, reg=0.1, W=w, calc_value=True, calc_grad_by_params=False)[0]
    softmax_grad_w = lambda w: demo_softmax.calc_value_and_grad(sample, label, reg=0.1, W=w, calc_value=False, calc_grad_by_params=True)[1]
    #gradient_test(softmax_val_w, softmax_grad_w, w, epsilon0=50, num_iter=30, delta=0.1)

    ##4. Test softmax Jacobian test ##

    softmax_val_w = lambda w: demo_softmax.calc_value_and_grad(sample, label, reg=0.1, W=w, calc_value=True, calc_grad_by_params=False)[0]
    softmax_grad_w = lambda w: demo_softmax.calc_value_and_grad(sample, label, reg=0.1, W=w, calc_value=False, calc_grad_by_params=True)[1]
    f_jacobianmv = lambda w, v: np.dot(softmax_grad_w(w), v.T)[0, 0]
    #jacobian_test(softmax_val_w, f_jacobianmv, w, epsilon0=50, num_iter=30, delta=0.1)


    ## Test softmax gradient w.r.t Params numerically ##

    #__, grad = demo_softmax.calc_value_and_grad(sample, label)
    #grad_err_1 = grad_check_full(softmax_val_w, demo_softmax.get_params_as_matrix(), grad, 10)
    #grad_err_2 = grad_check_sparse(softmax_val_w, demo_softmax.get_params_as_matrix(), grad, 10)
    #assert grad_err_2 < 0.01

    ###########  RESNET LAYER TESTS  ############


    demo_layer = ResLayer(dim=dim)
    sample = sample.T  # For the layers, the data should be transposed
    layer_w1= demo_layer.W1
    layer_w2=demo_layer.W2
    layer_b=demo_layer.b
    ##4. Test layer jacobian w.r.t X ##
    layer_val_x = lambda x: demo_layer.forward_pass(x)
    layer_jacobian_vec_x = lambda x, v: demo_layer.backward_pass(x, v)[-1]  # returns (dL\dX)^T . v
    #jacobian_test(layer_val_x, layer_jacobian_vec_x, sample, epsilon0=50, num_iter=30, delta=0.1)

    ##5. Test layer jacobian w.r.t W2 ##
    layer_val_w2 = lambda w2: demo_layer.forward_pass(sample,W2=w2)
    layer_jacobian_vec_w2 = lambda w2,v: demo_layer.backward_pass(sample, v,W2=w2)[-2]  # returns (dL\dX)^T . v
    jacobian_test(layer_val_w2, layer_jacobian_vec_w2,layer_w2, epsilon0=50, num_iter=30, delta=0.1,dim_d=sample.shape)

    ##6. Test layer jacobian w.r.t b ##
    layer_val_b = lambda b: demo_layer.forward_pass(sample,b=b)
    layer_jacobian_vec_b = lambda b, v: demo_layer.backward_pass(sample, v,b=b)[-2]  # returns (dL\dX)^T . v
    #jacobian_test(layer_val_b, layer_jacobian_vec_b, layer_b, epsilon0=50, num_iter=30, delta=0.1,dim_d=sample.shape)

    ##7. Test layer jacobian w.r.t w1 ##
    layer_val_w1 = lambda w1: demo_layer.forward_pass(sample,w1=w1)
    layer_jacobian_vec_w1 = lambda w1, v: demo_layer.backward_pass(sample, v,w1=w1)[-2]  # returns (dL\dX)^T . v
    #jacobian_test(layer_val_b, layer_jacobian_vec_b, layer_w1, epsilon0=50, num_iter=30, delta=0.1,dim_d=sample.shape)




    ## Test layer jacobian w.r.t Params ##

    ## TODO


def load_data(path):
    f = scipy.io.loadmat(path)
    t_data = f.get('Yt').T
    t_labels = np.argmax(f.get('Ct'), axis=0)
    v_data = f.get('Yv').T
    v_labels = np.argmax(f.get('Cv'), axis=0)
    num_labels = f.get('Cv').shape[0]

    return t_data, t_labels, v_data, v_labels, num_labels


def main(tests=False):

    PATH = os.getcwd()+'\datasets\GMMData.mat'
    t_data, t_labels, v_data, v_labels, num_labels = load_data(PATH)  # Data is of shape MxN where N is dim, M is # of samples

    # Normalize the data
    mean_image = np.mean(t_data, axis=0)
    t_data -= mean_image
    v_data -= mean_image
    t_data /= np.std(t_data, axis=0)
    v_data /= np.std(v_data, axis=0)

    if tests:
       run_tests(sample=v_data[None, 0], label=v_labels[None, 0], num_labels=num_labels)

    #print(np.std(t_data, axis=0), np.mean(t_data, axis=0))  # Expect variance = 1, mean = 0
"""
    hyperparams_grid = {
        "max_iter": [50],
        "batch_size": [200],
        "learning_rate": [0.001],
        "decay_rate": [0.1],
        "convergence_criteria": [0.01],
        "gamma": [0.8],
        "reg_param": [0.9],
        "num_layers": [2]
    }

    max_acc = 0
    cur_acc = 0
    best_params = {}

    for hp_comb in generate_all_combinations(hyperparams_grid):
        hyperparams = {key: hp_comb[i] for key, i in zip(list(hyperparams_grid.keys()), range(len(hyperparams_grid.keys())))}

        cur_acc = run_unit(t_data, t_labels, v_data, v_labels, num_labels, **hyperparams)
        if cur_acc > max_acc:
            best_params = hyperparams
            max_acc = cur_acc

    print("Maximal accuracy of %d on validation set, achieved with : %s" % (max_acc, str(best_params)))


def run_unit(t_data, t_labels, v_data, v_labels, num_labels, **hyperparams):

    print("Running unit with: " + str(hyperparams))
    # Hyper parameters
    MAX_ITER = hyperparams['max_iter']
    BATCH_SIZE = hyperparams['batch_size']
    LEARNING_RATE = hyperparams['learning_rate']
    DECAY_RATE = hyperparams['decay_rate']
    CONVERGENCE_CRITERIA = hyperparams['convergence_criteria']
    GAMMA = hyperparams['gamma']
    REG_PARAM = hyperparams['reg_param']
    NUM_LAYERS = 2
    if 'num_layers' in hyperparams.keys():
        NUM_LAYERS = hyperparams['num_layers']

    #model = Softmax(dim=t_data.shape[1], num_labels=num_labels, reg_param=REG_PARAM)

    model = ResNetwork(NUM_LAYERS, t_data.shape[1], REG_PARAM, num_labels)

    loss_history, accuracy_history = train_with_sgd(model, t_data=t_data, t_labels=t_labels, convergence_criteria=CONVERGENCE_CRITERIA,
                                                    decay_rate=DECAY_RATE,
                                                    batch_size=BATCH_SIZE,
                                                    max_iter=MAX_ITER,
                                                    learning_rate=LEARNING_RATE,
                                                    gamma=GAMMA,
                                                    v_data=v_data,
                                                    v_labels=v_labels)

    #print(loss_history)
    # predictions = sm.predict(v_data)
    iterations = list(range(0, len(accuracy_history['test_set'])))
    test_accs = accuracy_history['test_set']
    validation_accs = accuracy_history['validation_set']

    plt.subplot(2, 1, 1)
    plt.plot(iterations, test_accs, 'o-')
    plt.title('Accuracies')
    plt.ylabel('test accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(iterations, validation_accs, '.-')
    plt.xlabel('iteration')
    plt.ylabel('validation accuracy')

    plt.show()

    return validation_accs[-1]
"""

if __name__ == "__main__":
    main(tests=TEST_MODE)
