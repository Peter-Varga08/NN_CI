
""" Assignment_1 of Neural Networks and Computational Intelligence:
 Implementation of Rosenblatt Perceptron Algorithm
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#### Define parameters for experiments
N = [20,40]  # number of features for each datapoint
alpha = [x / 100 for x in range(75, 325, 15)]  # ratio of (datapoint_amount / feature_amount)
n_D = 50  # number of datasets required for each value of P
n_max = 100  # maximum number of epochs

#### Params for data generation
MU = 0
SIGMA = 1

def generate_data(n: int, p: int) -> tuple([np.ndarray, np.ndarray]):
    """ Generation of artificial dataset containing P randomly generated N-dimensional feature vectors and labels.
        - The datapoints are sampled from a Gaussian distribution with mu=0 and std=1.
        - The labels are independent random numbers y = {-1,1}.
    :return: Generated dataset as a PxN numpy array; labels as Px1 numpy array.
    """
    X = []
    for i in range(n):
        X.append(np.random.normal(MU, SIGMA, p))

    X_clamped = X.copy()
    X_clamped.append(-1 * np.ones(p))
    X = np.asarray(X).transpose()
    X_clamped = np.asarray(X_clamped).transpose()

    y = np.asarray([np.random.choice([-1, 1]) for _ in range(len(X))])
    return X, y, X_clamped

def train(n: int, p: int, epochs: int, data: np.ndarray, labels: np.ndarray):
    """ Implementation of sequential perceptron training by cyclic representation of the P examples.
     :param n: Number of features. A single chosen value from N. E.g.: N[0].
     :param p: Number of examples. A single chosen value from P. E.g.: P[0][0]. First index has to match index of N.
     :param epochs: Number of epochs.
     :param data: Dataset containing generated examples using 'generate_data' funct.
     :return: w : the final weights of the perceptron after training
              i : the number of epochs reached 
     """
    w = np.zeros(len(data[0,:]))
    for i in range(epochs):
        weight_step_taken = False #This tracks whether a step has been taken in the current epoch
        for j in range(p):
            #calculating E^mu
            E_mu = np.dot(w, data[j,:]) * labels[j]

            #perform weight update
            if E_mu <= 0: 
                weight_step_taken = True
                w += (1/n) * data[j,:] * labels[j]

        #this breaks the outer training loop if no weight step was performed in an entire epoch
        if weight_step_taken == False:
            break

    return w, i #also returning i as its important for determining if a solution was found


# CREATING DATASETS
def generate_data_dict(N) -> dict:
    """ -- This generates the dict of datasets where the first key is the number of features, 
    the keys inside those are the number of datapoints within each dataset, e.g.:
    datasets[20][35] means you are accessing a dataset which has 20 features and 35 datapoints.
        -- The actual array which contains the values for each datapoint in the previous example can be selected via
    datasets[20][35][0][0], the labels array can be selected via datasets[20][35][0][1]. This is because there are
    n_D = 50 datasets for each configuration, thus len(datasets[20][35])) would output 50 for example.
    """

    datasets = {}
    for i in range(len(N)):
        datasets[N[i]] = {} # datasets with 20 and 40 features

    for n in N:
        P = [int(a * n) for a in alpha]  # number of datapoints per dataset FOR the current N
        for p in P:
            datasets[n][p] = []
            for _ in range(n_D):  # create n_D datasets for each P
                datasets[n][p].append(generate_data(n, p))
    return datasets 

if __name__ == '__main__':
    proportion_successful = np.zeros((len(N), len(alpha)))
    proportion_successful_clamped = np.zeros((len(N), len(alpha)))
    datasets = generate_data_dict(N)

    #calculating the proportion of successful perceptron training runs
    for m in tqdm(range(len(N))):
        for k in tqdm(range(len(alpha))):
            p = int(alpha[k] * N[m])
            number_successful = 0
            number_successful_clamped = 0
            for _ in range(n_D):
                w_final, i = train(N[m], p, n_max, datasets[N[m]][p][_][0], datasets[N[m]][p][_][1])
                w_final_clamped, i_clamped = train(N[m], p, n_max, datasets[N[m]][p][_][2], datasets[N[m]][p][_][1])
                if i < n_max - 1:
                    number_successful += 1
                if i_clamped < n_max - 1:
                    number_successful_clamped += 1
            proportion_successful[m, k] = number_successful/n_D
            proportion_successful_clamped[m, k] = number_successful_clamped/n_D
    print(w_final)
    print(w_final_clamped)

fig, ax = plt.subplots(nrows=2, ncols=1)

a = 0
for row in ax:
    if a == 0: 
        row.plot(alpha ,proportion_successful[0,:], label = f'N = {N[0]}')
        row.plot(alpha ,proportion_successful_clamped[0,:], label = f'N = {N[0]},  clamped')
        row.legend()
        row.set_ylabel('proportion_successful')
        row.set_xlabel('alpha')
        a = 1 
    else:
        row.plot(alpha ,proportion_successful[1,:], label = f'N = {N[1]}')
        row.plot(alpha ,proportion_successful_clamped[1,:], label = f'N = {N[1]},  clamped')
        row.legend()
        row.set_ylabel('proportion_successful')
        row.set_xlabel('alpha')
plt.show()
    




# TODO: Extensions
# 1) Observe behaviour of Q_l.s.
# 2) Determine embedding strengths x^mu
# 3) Use non-zero value for 'c'
# 4) Inhomogeneous perceptron with clamped inputs
# 5) ...