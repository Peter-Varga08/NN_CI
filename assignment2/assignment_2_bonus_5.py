""" Assignment_2 of Neural Networks and Computational Intelligence:
 Implementation of Learning a linearly separable rule
"""

from typing import List
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pprint
import random

np.random.seed(42)
random.seed(42)

#### Define parameters for experiments
N = [20, 40, 60, 80]  # number of features for each datapoint
alpha = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]  # ratio of (datapoint_amount / feature_amount)
n_D = 10  # number of datasets required for each value of P
n_max = 100  # maximum number of epochs

#### Params for data generation
MU = 0
SIGMA = 1


def generate_wstar(n: list) -> list:
    """ Generation of teacher perceptron, that is, a randomly drawn w* with |w*|^2 = N.
    The above equation means that we are looking for (w_1)^2 + (w_2)^2 ... + (w_n)^2 being equal to N, thus the
    problem can be solved with the following:
    - The idea is to generate N-1 points on an interval of [0, N], then take the length of the intervals (determined
    by the sorted order of generated numbers) as the random numbers themselves,
    with the last number being the remainder interval.
    Lastly, take the sqrt of the individual numbers as the entries of the desired teacher vector w*.

    :param n: list of number of features.
    :return: w_star : The final weights of the teacher perceptron.
    """
    w_star = []
    for num in n:
        points = sorted(np.random.uniform(0, num, num - 1))
        intervals = [points[i] - points[i - 1] for i in range(1, len(points))]
        intervals.insert(0, points[0])  # first interval is the first value (e.g. x1-0)
        intervals.append(num - points[-1])  # last interval is N-x[-1]
        w_star.append(np.array([np.sqrt(i) for i in intervals]))
    return w_star


def generate_data(n: int, p: int, w_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Generation of artificial dataset containing P randomly generated N-dimensional feature vectors and labels.
        - The datapoints are sampled from a Gaussian distribution with mu=0 and std=1.
        - The labels are the outputs of the teacher perceptron.
    :return: Generated dataset as a PxN numpy array; labels as Px1 numpy array.
    """
    X = []
    for i in range(n):
        X.append(np.random.normal(MU, SIGMA, p))
    X = np.asarray(X).transpose()
    # |--------------------------------|
    # |   EXTENSION 5: Noisy examples  |
    # |--------------------------------|
    LAMBDA = 0.1
    choices = random.choices([1, -1], weights=[1 - LAMBDA, LAMBDA], k=len(X))
    y = np.asarray(
        [np.sign(np.dot(w_star, x)) if choices[idx] == 1 else -np.sign(np.dot(w_star, x)) for idx, x in enumerate(X)])
    return X, y


def train(n: int, p: int, epochs: int, data: np.ndarray, labels: np.ndarray, w_star: np.ndarray):
    """ Implementation of sequential perceptron training by cyclic representation of the P examples.
     :param n: Number of features. A single chosen value from N. E.g.: N[0].
     :param p: Number of examples. A single chosen value from P. E.g.: P[0][0]. First index has to match index of N.
     :param epochs: Number of epochs.
     :param data: Dataset containing generated examples using 'generate_data' funct.
     :param labels: Labels of each example from 'data'.
     :param w_star: An n-dim teacher perceptron.
     :return: w : the final weights of the perceptron after training
              i : the number of epochs reached
     """

    w = np.zeros(n)

    # The new loop goes until t_max = n_max * p single training steps have been performed
    for i in range(epochs * p):
        # |-------------------|
        # | Minover algorithm |
        # |-------------------|
        # 1) determine stabilities of all examples at each step
        kappas = []
        for k in range(p):
            kappas.append(np.dot(w, data[k, :] * labels[k]))  # we don't need to divide by the norm of w(t)
        # 2) identify example mu(t) that has currently the minimal stability
        mu_min = kappas.index(min(kappas))
        # 3) perform a Hebbian update step with it
        w += (1 / n) * data[mu_min, :] * labels[mu_min]

    # check for cosine criterion in at the end of training
    similarity = np.dot(w, w_star) / (np.linalg.norm(w) * np.linalg.norm(w_star))
    # determine generalization error
    error = 1 / np.pi * np.arccos((np.dot(w, w_star)) / (np.linalg.norm(w) * np.linalg.norm(w_star)))

    return w, i, error, similarity


# CREATING DATASETS
def generate_data_dict(N: List[int], w_stars: List[np.ndarray]) -> dict:
    """ -- This generates the dict of datasets where the first key is the number of features,
    the keys inside those are the number of datapoints within each dataset, e.g.:
    datasets[20][35] means you are accessing a dataset which has 20 features and 35 datapoints.
        -- The actual array which contains the values for each datapoint in the previous example can be selected via
    datasets[20][35][0][0], the labels array can be selected via datasets[20][35][0][1]. This is because there are
    n_D = 50 datasets for each configuration, thus len(datasets[20][35]) would output 50 for example.
    """

    datasets = {}
    for i in range(len(N)):
        datasets[N[i]] = {}

    for idx, n in enumerate(N):
        P = [int(a * n) for a in alpha]  # number of datapoints per dataset FOR the current N
        for p in P:
            datasets[n][p] = []
            for _ in range(n_D):  # create n_D datasets for each P
                datasets[n][p].append(generate_data(n, p, w_stars[idx]))
    return datasets


if __name__ == '__main__':
    errors = np.zeros((len(N), len(alpha)))  # we collect errors instead of successful runs this time
    perceptrons = generate_wstar(N)  # an array of perceptrons
    datasets = generate_data_dict(N, perceptrons)

    # calculating the proportion of successful perceptron training runs
    experiment_params = {n: {a: [0, 0] for a in alpha} for n in N}
    for m in tqdm(range(len(N))):
        for k in tqdm(range(len(alpha))):
            p = int(alpha[k] * N[m])
            training_steps = []
            cos_sim = []
            curr_errors = []  # generalization errors of the next 10 datasets
            for _ in range(n_D):
                X, y = datasets[N[m]][p][_][0], datasets[N[m]][p][_][1]
                w_final, i, e, similarity = train(N[m], p, n_max, X, y, perceptrons[m])
                curr_errors.append(e)
                training_steps.append(i + 1)
                cos_sim.append(similarity)
            errors[m, k] = np.mean(curr_errors)
            experiment_params[N[m]][alpha[k]][0] = np.mean(training_steps)
            experiment_params[N[m]][alpha[k]][1] = np.mean(cos_sim)
    print("Experiment params", experiment_params)
    plt.figure()
    for i in range(len(N)):
        plt.plot(alpha, errors[i, :], label=f'N = {N[i]}')
    plt.xlabel('alpha')
    plt.ylabel('Generalization error')
    plt.legend()
    plt.show()
