""" Assignment_1 of Neural Networks and Computational Intelligence:
 Implementation of Rosenblatt Perceptron Algorithm
"""

import numpy as np

np.random.seed(42)

#### Define parameters for experiments
N = [20, 40]  # number of features for each datapoint
alpha = [x / 100 for x in range(75, 325, 25)]  # ratio of (datapoint_amount / feature_amount)
n_D = 50  # number of datasets required for each value of P
n_max = 100  # maximum number of epochs

#### Params for data generation
MU = 0
SIGMA = 1


def generate_data(n: int, p: int) -> [np.ndarray, np.ndarray]:
    """ Generation of artificial dataset containing P randomly generated N-dimensional feature vectors and labels.
        - The datapoints are sampled from a Gaussian distribution with mu=0 and std=1.
        - The labels are independent random numbers y = {-1,1}.
    :return: Generated dataset as a PxN numpy array; labels as Px1 numpy array.
    """
    X = []
    for i in range(n):
        X.append(np.random.normal(MU, SIGMA, p))
    X = np.asarray(X).transpose()

    y = np.asarray([np.random.choice([-1, 1]) for _ in range(len(X))])
    return X, y


def perceptron_algorithm(X: np.ndarray, y: np.ndarray, w: int, n: int) -> [np.ndarray, bool]:
    """ Implementation of the Rosenblatt Perceptron algorithm.
    :return: Updated weight vector and boolean to indicate whether weight vector was updated or not.
    """

    E_mu = np.dot(w, X * y)
    if E_mu <= 0:
        w_new = w + (1/n) * X * y
        is_w_changed = True
    else:
        w_new = w
        is_w_changed = False

    return w_new, is_w_changed


def train(n: int, epochs: int, data: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Implementation of sequential perceptron training by cyclic representation of the P examples.
     :param n: Number of features. A single chosen value from N. E.g.: N[0].
     :param epochs: Number of epochs.
     :param data: Dataset containing generated examples using 'generate_data' funct.
     :return: Weight vector array and E_mu array (for all epochs), and epoch number at which training stopped.
     """
    assert epochs < n_max+1, "Epoch number error, can't be higher than n_max"

    # - w: weight vector, where "w(t) = weight at timestep t", thus we store all weights that have occurred
    # - len(data)+1 means we are going to have 1 more value in the weight vector than datapoint,
    #   because first value is always w(0) = 0
    len_p = len(data[0])
    w = np.zeros((epochs, len_p+1, n))
    E_mu = np.zeros((epochs, len_p))
    for e in range(epochs):
        for i in range(len_p):
            X, y = data[0][i], data[1][i]  # x∈R^n, y∈R
            w[e][i+1], E_mu[e][i] = perceptron_algorithm(X, y, w[e][i], n)
    # Training is performed until solution is found, such that E_mu > 0 for all mu, or max number of sweeps is reached
        if not any(E_mu[e]):
            print(f"Solution has been found in epoch {e}")
            return w, E_mu, e, True
        # At the end of the current epoch, assign w(t), where t=len_p+1, as w(0) for next epoch
        if e < epochs-1:
            w[e+1][0] = w[e][-1]
    print("Training has ended due to reach of maximum number of sweeps, solution has not been found.")
    return w, E_mu, epochs, False


# CREATING DATASETS
datasets = {20: {}, 40: {}}  # datasets with 20 and 40 features
for n in N:
    P = [int(a * n) for a in alpha]  # number of datapoints per dataset FOR the current N
    for p in P:
        datasets[n][p] = []
        for _ in range(n_D):  # create n_D datasets for each P
            datasets[n][p].append(generate_data(n, p))

# RUN TRAINING for a single dataset (just for testing purpose)
f = 20  # number of features
d = 30  # number of datapoints
max_epochs = 100  # number of maximum epochs to perform the training for
w, E_mu, epoch_count, is_solution = train(f, max_epochs, datasets[f][d][0])

# RUN TRAINING for ALL datasets
sr_count = 0  # count of successful runs
alpha_success = {a: 0 for a in alpha}
for n in datasets.keys():
    for idx, p in enumerate(datasets[n].keys()):
        for dataset in datasets[n][p]:
            _, _, _, is_solution = train(n, max_epochs, dataset)
            if is_solution:
                sr_count += 1
                a = p/n
                alpha_success[a] += 1

# TODO: Extensions
# 1) Observe behaviour of Q_l.s.
# 2) Determine embedding strengths x^mu
# 3) Use non-zero value for 'c'
# 4) Inhomogeneous perceptron with clamped inputs
# 5) ...
