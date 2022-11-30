import numpy as np
import math

NUM_CLASSIFIERS = 2


def create_individual():
    return np.random.random(NUM_CLASSIFIERS*2).tolist()


def calc_new_sigma(new_fitness, old_fitness, sigma):
    #print(f"calculating new sigma from {sigma}, for {new_fitness}, after {old_fitness}")
    fitness_delta = new_fitness - old_fitness
    theta = 0
    if fitness_delta > 0:
        theta = 0.5
    if fitness_delta < 0:
        theta = -1
    print(f"fitness delta = {fitness_delta}, theta = {theta}")

    tau_global = 1 / math.sqrt(2 * NUM_CLASSIFIERS)
    tau_local = 1 / math.sqrt(2 * math.sqrt(NUM_CLASSIFIERS))
    e_exp = theta * (tau_global * np.random.normal(0.0, 1.0)) + (tau_local * np.random.normal(0.0, 1.0))
    new_sigma = sigma * math.e ** e_exp
    return new_sigma


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sigmas = [np.array([1.0, 1.0])]
    fitnesses = list(range(5, 100, 5))
    for i in range(1,len(fitnesses)):
        sigmas += [calc_new_sigma(fitnesses[i], fitnesses[i-1], sigmas[i-1])]

    for i in range(len(fitnesses)):
        print(f"{fitnesses[i]}\t{sigmas[i].tolist:.3f}")

