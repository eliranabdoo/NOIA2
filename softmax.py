import numpy as np


class LossFunction:
    def calc_value_and_grad(self):
        pass

    # Should always be able to update based on the same format it outputs the params in get_params
    def update_params(self, P):
        pass

    def get_params(self):
        pass


class ResNetwork(LossFunction):
    def __init__(self, L, dim, reg_param, num_labels):
        self.dim = dim
        self.L = L
        self.res_layers = [ResLayer(dim) for i in range(0, self.L)]
        self.softmax = Softmax(dim=dim, num_labels=num_labels, reg_param=reg_param)


    def calc_value_and_grad(self, X, y, reg=None, P=None, calc_value=True, calc_grad_by_params=True):
        """Returns the gradients with only respect to the parameters (without x)"""

        if P is not None:
            self.update_params(P)
        if reg is not None:
            self.softmax.reg=reg

        gradient, loss = None, None
        num_of_samples = X.shape[0]
        sum_of_gradients = None
        sum_of_losses = 0

        for i in range(0, num_of_samples):
            sample = X[None, i]  # sample should be kept 1xN
            label = y[None, i]  # labels should be kept as vector
            if calc_value:
                x_history = self.forward_pass(sample, label)
                sum_of_losses += x_history[-1]

            if calc_grad_by_params:
                cur_gradient = self.flatten_and_append(self.backward_pass(label, x_history))
                if sum_of_gradients is None:
                    sum_of_gradients = cur_gradient
                else:
                    sum_of_gradients += cur_gradient

        if calc_grad_by_params:
            gradient = sum_of_gradients / num_of_samples
        if calc_value:
            loss = sum_of_losses / num_of_samples
        return loss, gradient

    def backward_pass(self, y, x_history):  #X is a sample
        """

        :param y:
        :param x_history: assumes x_history is adapted to the softmax and layer's input sizes, as built in forward_pass
        :return: list of the params (mat by mat)
        """
        result = []
        softmax_input = x_history[-2]
        __, softmax_gradient = self.softmax.calc_value_and_grad(softmax_input, y, calc_value=False)
        v = self.softmax.calc_grad_by_x(softmax_input, y)  # No transpose is needed as it seems
        result = [softmax_gradient] + result  # perhaps transpose is needed

        for i in range(1, self.L+1):
            current_layer = self.res_layers[self.L-i]
            input = x_history[-2-i]
            dw1, db, dw2, dx = current_layer.backward_pass(input, v)
            v = dx
            result = [dw1, db, dw2] + result

        return result

    def forward_pass(self, X, y):  #X is a sample
        """

        :param X: MxN
        :param y: (M, ) vector
        :param P: Params vector (can be used with update_params)
        :param reg: scalar
        :return:
        """

        cur_x = X.T  # We transpose since its a layer's input
        x_history = []
        x_history.append(cur_x)

        for i in range(0,self.L):
            cur_x = self.res_layers[i].forward_pass(x_history[-1])  # We transpose when interacting with layer
            x_history.append(cur_x)

        x_history[-1] = x_history[-1].T  # We transpose since the last layer's output is the softmax's input

        if y is not None:
            loss, __ = self.softmax.calc_value_and_grad(cur_x.T, y, calc_grad_by_params=False)  # We again transpose since output of layer goes to softmax
            x_history.append(loss)

        return x_history

    def update_params(self, P):
        """ Gets the whole vector from SGD, and need to update each of its layers properly
        length is L*(2N**2 + N) + (N+1)*num_labels
        <W11|B1|W21|W12|B2|W22|...|WSMBSM>"""
        N = self.dim
        N2 = N**2
        l = self.softmax.num_labels
        L = self.L
        num_params_in_layer = 2*(N2) + N
        num_params_in_softmax = (N+1)*l

        params_consumed = 0
        i = 0

        for layer in self.res_layers:
            cur_params_vec = P[i*num_params_in_layer:(i+1)*num_params_in_layer]
            cur_W1 = cur_params_vec[:N2].reshape(N, N)
            cur_b = cur_params_vec[N2:N2+N].reshape(N, 1)
            cur_W2 = cur_params_vec[N2+N:].reshape(N,N)
            layer.update_params(cur_W1, cur_b, cur_W2)
            params_consumed += num_params_in_layer
            i += 1

        assert len(P) - params_consumed == num_params_in_softmax

        W = P[-num_params_in_softmax:].reshape(N+1, l)
        self.softmax.update_params(W)

    def get_params(self):
        """ Gets the whole network's parameter as a vector of size L(2N^2+N)+(N+1)*num_labels"""
        params_list = []
        for layer in self.res_layers:
            params_list.extend(layer.get_params())
        params_list.append(self.softmax.get_params())
        return self.flatten_and_append(params_list)

    def flatten_and_append(self, params_list):
        res = []
        for param in params_list:
            res = np.append(res, param.flatten())
        return res

    def predict(self, X):
        x_history = self.forward_pass(X, y=None)
        return np.argmax(self.softmax.add_bias_dimension(x_history[-1]).dot(self.softmax.get_params()), axis=1)


class ResLayer:
    def __init__(self, dim):
        self.N = dim # dimensionality
        self.W1 = np.random.randn(self.N, self.N)*np.sqrt(2/self.N)  # NxN mat
        self.W2 = np.random.randn(self.N, self.N)*np.sqrt(2/self.N)  # NxN mat
        self.b = np.zeros([self.N, 1], dtype=np.double)  # Nx1  col vec

    def forward_pass(self, X, W1=None, W2=None, b=None):
        """

        :param X: NxM
        :param W1: NxN
        :param W2: NxN
        :param b: Nx1
        :return:
        """
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
        if b is None:
            b = self.b
        return X + self.W2.dot((self.W1.dot(X).T + self.b.T).T)

    def backward_pass(self, X, v, W1=None, W2=None, b=None):
        """

        :param X: NxM
        :param v: NxM vector
        :param W1: NxN
        :param W2: NxN
        :param b: Nx1
        :return:
        """
        if W1 is None:
            W1 = self.W1
        if W2 is None:
            W2 = self.W2
        if b is None:
            b = self.b
        W1_d_x_p_b = np.add(W1.dot(X), b)
        sig = sigmoid(W1_d_x_p_b)
        sig_derivative = np.multiply(sig, 1 - sig)
        dy_db = np.multiply(W2, sig_derivative.T)
        dy_db_t_v = (dy_db).T.dot(v)
        dy_dw2_t_v = v.dot(sig.T)
        dy_dw1_t_v = dy_db_t_v.dot(X.T)
        new_v = (np.eye(self.N, self.N) + dy_db.dot(W1)).T.dot(v)
        return dy_dw1_t_v, dy_db_t_v, dy_dw2_t_v, new_v

    def update_params(self, W1, b, W2):
        self.W1 = W1
        self.b = b
        self.W2 = W2

    def get_params(self):
        return self.W1, self.b, self.W2


class Softmax(LossFunction):
    def __init__(self, dim=None, num_labels=None, reg_param=None):
        self.W = np.random.randn(dim, num_labels)*np.sqrt(2/(dim+1))
        self.b = np.zeros([1, num_labels], dtype=np.double)
        self.reg = reg_param
        self.num_labels = num_labels

    def add_bias_dimension(self, X):
        return np.column_stack((X, np.ones(X.shape[0])))

    def calc_value_and_grad(self, X, y, reg=None, W=None, calc_value=True, calc_grad_by_params=True):
        """

        :param X: MxN
        :param y: (M,) vector
        :param reg: scalar
        :param W: (N+1)xNUM_LABELS
        :return:
        """

        if W is None:
            W = self.get_params()
        if reg is None:
            reg=self.reg
        X = self.add_bias_dimension(X)

        loss, dW = None, None

        num_classes = W.shape[1]
        num_train = X.shape[0]

        scores = X.dot(W)
        scores_exp = np.exp(scores)

        numerical_stab_factors = np.max(scores, axis=1)
        normalized_scores = np.exp(scores.T - numerical_stab_factors.T).T
        scores_sums = np.sum(normalized_scores, axis=1)

        if calc_grad_by_params:
            total_scores_mat = (normalized_scores.T / scores_sums.T).T
            labels_mat = np.zeros_like(scores)
            labels_mat[np.arange(0, num_train), y] = 1
            dW = np.zeros_like(W)
            dW += (X.T).dot(total_scores_mat - labels_mat)
            dW /= num_train
            dW += (2 * reg) * W

        if calc_value:
            class_scores = normalized_scores[np.arange(len(scores_exp)), y.T]
            loss = 0.0
            loss = np.sum(np.log(scores_sums) + np.log(np.ones(num_train) / class_scores))
            loss /= num_train
            loss += reg * np.sum(W * W)

        return loss, dW

    def calc_grad_by_x(self, X, y, reg=None, W=None):
        """

        :param W:(N+1)xNUM_LABELS
        :param X: MxN
        :param y: (M,) vector
        :return:
        """

        if W is None:
            W = self.get_params()
        if reg is None:
            reg=self.reg
        X = self.add_bias_dimension(X)

        num_classes = W.shape[1]
        num_train = X.shape[0]

        scores = X.dot(W)
        scores_exp = np.exp(scores)

        numerical_stab_factors = np.max(scores, axis=1)
        normalized_scores = np.exp(scores.T - numerical_stab_factors.T).T
        scores_sums = np.sum(normalized_scores, axis=1)
        total_scores_mat = (normalized_scores.T / scores_sums.T).T
        labels_mat = np.zeros_like(scores)
        labels_mat[np.arange(0, num_train), y] = 1

        return W.dot((total_scores_mat - labels_mat).T)[:-1]  # We cut the last element

    def get_params(self):
        return np.vstack((self.W, self.b))

    def update_params(self, P):
        self.b = P[-1]
        self.W = P[0:-1]

    def predict(self, X):
        return np.argmax(self.add_bias_dimension(X).dot(self.get_params()), axis=1)




def accuracy(predictions, labels):
    return (100.0 * np.sum(predictions ==labels)
          / predictions.shape[0])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def update_learning_rate_simple(learning_rate, decay_rate, iteration):
    if iteration < 100:
        return 0.01
    else:
        return 0.001


def update_learning_rate_step(initial_learning_rate, interval, iteration, drop_rate):
    return initial_learning_rate * np.power(drop_rate,
                                     np.floor((1 + iteration) / interval))


def train_with_sgd(loss_function, t_data, t_labels, max_iter, learning_rate, decay_rate,
                   batch_size, convergence_criteria, gamma ,v_data, v_labels):
    """
    We assume that the function can receive dynamic data size
    :param loss_function:
    :param t_data: The data set (ideally should be loaded to RAM on demand)
    :param t_labels: The corresponding labels
    :param max_iter:
    :param learning_rate:
    :param decay_rate:
    :param batch_size:
    :return:
    """
    m = np.zeros(loss_function.get_params().shape, dtype=np.double)
    loss_history = []
    accuracy_history = {"test_set": [],
                        "validation_set": []
                        }
    cur_learning_rate = learning_rate
    num_train, dim = t_data.shape
    num_of_batches = int(np.ceil(num_train / batch_size))
    cur_loss = 0.0
    for i in range(0, max_iter):
        x_batch = None
        y_batch = None

        cur_learning_rate = update_learning_rate_step(learning_rate, 10, i, decay_rate)

        assert len(t_data) == len(t_labels)
        p = np.random.permutation(num_train)
        t_data = t_data[p]
        t_labels = t_labels[p]
        for j in range(0, num_of_batches):
            x_batch = t_data[j * batch_size:(j + 1) * batch_size]
            y_batch = t_labels[j * batch_size:(j + 1) * batch_size]

            cur_loss, grad = loss_function.calc_value_and_grad(x_batch, y_batch, calc_value=True, calc_grad_by_params=True)

            prev_params = loss_function.get_params()

            # Momentum update
            m = gamma * m + cur_learning_rate*grad
            updated_params = prev_params - m

            loss_function.update_params(updated_params)
            loss_history.append(cur_loss)

        test_set_accuracy = accuracy(loss_function.predict(t_data), t_labels)
        validation_set_accuracy = accuracy(loss_function.predict(v_data), v_labels)

        print("After %d epochs, train set accuracy is %d" % (i+1, test_set_accuracy))
        print("After %d epochs, validation set accuracy is %d" % (i+1, validation_set_accuracy))

        accuracy_history['test_set'].append(test_set_accuracy)
        accuracy_history['validation_set'].append(validation_set_accuracy)

        #if np.abs(loss_history[-1]-loss_history[-2]) < convergence_criteria:  ## TODO
        #    break

    return loss_history, accuracy_history
