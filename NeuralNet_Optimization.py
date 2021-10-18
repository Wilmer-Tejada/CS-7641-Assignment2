# 1. Learning curves
# 2. Loss per iteration
# 3. How many iterations to convergence / wall clock time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import mlrose_hiive as mlrose
import time

# load our dataset
df = pd.read_csv("Assignment2/data/breast-cancer.csv")
df = df.drop(labels=["Unnamed: 32"],axis=1)

# Recode y values to numeric.
df["diagnosis"] = df["diagnosis"].replace("M",1)
df["diagnosis"] = df["diagnosis"].replace("B",0)

# Seperate X and y columns from df
X = df.drop('diagnosis', axis = 1).values
y = df['diagnosis'].values


# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Load Scaler
scaler = StandardScaler()

# Scale data after splitting to avoid data leakage.
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

################################################# Build NNs ############################################################
nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[40],
                                    activation='relu',
                                    algorithm='random_hill_climb',
                                    max_iters=1000,
                                    bias=True,
                                    is_classifier=True,
                                    learning_rate=1,
                                    early_stopping=False,
                                    clip_max=5,
                                    max_attempts=100,
                                    random_state=42,
                                    curve = True,
                                    restarts=0
                                    )
nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[40],
                                   activation='relu',
                                   algorithm='simulated_annealing',
                                   max_iters=1000,
                                   bias=True,
                                   is_classifier=True,
                                   learning_rate=.9,
                                   early_stopping=True,
                                   clip_max=5,
                                   max_attempts=1000,
                                   random_state=42,
                                   schedule = mlrose.ExpDecay(),
                                   curve = True)
nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[40],
                                   activation='relu',
                                   algorithm='genetic_alg',
                                   max_iters=40,
                                   bias=True,
                                   is_classifier=True,
                                   learning_rate=.1,
                                   early_stopping=False,
                                   clip_max=5,
                                   max_attempts=100,
                                   random_state=42,
                                   curve=True,
                                   pop_size = 40,
                                   mutation_prob = .5)

############################# Grad Desc Fit ################################################
t = time.time()
nn_model = MLPClassifier(solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(5, 2), random_state=42, max_iter = 1000)
nn_model.fit(X_train, y_train)
train_time_gd = time.time() - t
############################# RHC Fit ################################################
t = time.time()
nn_model_rhc.fit(X_train, y_train)
rhc_fitness = nn_model_rhc.fitness_curve
train_time_rhc = time.time() - t
############################# SA Fit ################################################
t = time.time()
nn_model_sa.fit(X_train, y_train)
sa_fitness = nn_model_sa.fitness_curve
train_time_sa = time.time() - t
############################# GA Fit ################################################
t = time.time()
nn_model_ga.fit(X_train, y_train)
ga_fitness = nn_model_ga.fitness_curve
train_time_ga = time.time() - t

############################# Loss Plot ################################################
plt.close()
plt.plot(rhc_fitness[:,1],rhc_fitness[:,0], label = "RHC")
plt.plot(sa_fitness[:,1],sa_fitness[:,0], label = "SA")
plt.plot(ga_fitness[:,1],ga_fitness[:,0], label = "GA")
# plt.plot(gd_fitness[:,1],gd_fitness[:,0], label = "GD")

plt.title("Neaural Nets")
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.savefig("Assignment2/charts/Neural Nets")
plt.show()

############################# Time Plot ################################################
plt.close()
plt.bar("Gradient Descent",train_time_gd)
plt.bar("RHC",train_time_rhc)
plt.bar("SA",train_time_sa)
plt.bar("GA",train_time_ga)

plt.title("Neural Network Train Time")
plt.xlabel('Algorithm')
plt.ylabel('Train Time')
plt.savefig("Assignment2/charts/Neural Nets Train Time")
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, scoring="accuracy",
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # plot scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.title(title)

    return plt


###### GD Learning Curve
plt.close()
title = "Learning Curve  (Gradient Descent)"



plot_learning_curve(nn_model, title, X_train, y_train, cv=5)
plt.xlabel("Training Examples Used")
plt.ylabel("Accuracy")
plt.savefig('Assignment2/charts/NeuralNets_GD_Learning_Curve')
plt.show()

###### RHC Learning Curve
plt.close()
title = "Learning Curve  (RHC)"
plot_learning_curve(nn_model_rhc, title, X_train, y_train, cv=5)
plt.xlabel("Training Examples Used")
plt.ylabel("Accuracy")
plt.savefig('Assignment2/charts/NeuralNets_RHC_Learning_Curve')

plt.show()

###### SA Learning Curve
plt.close()
title = "Learning Curve  (SA)"
plot_learning_curve(nn_model_sa, title, X_train, y_train, cv=5)
plt.xlabel("Training Examples Used")
plt.ylabel("Accuracy")
plt.savefig('Assignment2/charts/NeuralNets_SA_Learning_Curve')

plt.show()

###### GA Learning Curve
plt.close()
title = "Learning Curve  (GA)"
plot_learning_curve(nn_model_ga, title, X_train, y_train, cv=5)
plt.xlabel("Training Examples Used")
plt.ylabel("Accuracy")
plt.savefig('Assignment2/charts/NeuralNets_GA_Learning_Curve')

plt.show()

