import numpy as np
from EKFalgorithm import EKF
from deap import base, creator, tools, algorithms
from scipy.stats import poisson  # check if poisson actually makes a difference or if bernoulli should be used
from bitstring import BitArray


def train_evaluate(ga_individual_sol):
    gene_length = 5
    R = ga_individual_sol[0:gene_length]
    P = ga_individual_sol[gene_length:2*gene_length]
    Q = ga_individual_sol[2*gene_length:3*gene_length]
    R = BitArray(R).uint
    P = BitArray(P).uint
    Q = BitArray(Q).uint

    # resize hyperparameterss to be within range
    R = R/1000
    P = np.interp(P, [0, 31], [0.00001, 0.1])
    Q = np.interp(Q, [0, 31], [0.00001, 0.1])

    print(f'R = {R}, P = {P}, Q = {Q}')

    # Run EKF
    try:
        soc_est, vt_est, vt_err, vt_act, total_err = EKF(R, P, Q, 'B0005')
    except np.linalg.LinAlgError:
        total_err = 1000000

    return [total_err]


if __name__ == '__main__':  
    # init variables of ga using deap
    # set seed for reproducibility
    np.random.seed(7)
    # set number of generations
    ngen = 10
    # set population size
    popsize = 10
    # set gene length
    gene_length = 5
    entire_length = 3*gene_length

    # basically creates classes for the fitness and individual
    creator.create('FitnessMax', base.Fitness, weights=[-1.0])
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # create toolbox & initialize population (bernoulli random variables)
    toolbox = base.Toolbox()
    toolbox.register("binary", poisson.rvs, 0.5) 
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=entire_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ordered cross over for mating
    toolbox.register("mate", tools.cxOrdered)
    # mutati
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)
    # selection algorithm
    toolbox.register("select", tools.selTournament, tournsize=int(popsize/2))
    # evaluation fitness of individuals
    toolbox.register("evaluate", train_evaluate)  # this train evaluate might not be allowed to have gene)length as input

    population = toolbox.population(n=popsize)
    r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.3, ngen=ngen, verbose=True)

    # print best solution found
    best_individuals = tools.selBest(population, k=2)
    print('Best ever individual = ', best_individuals[0], '\nFitness = ', best_individuals[0].fitness.values[0])
    print(f'list of individuals = {best_individuals}')
