import numpy as np
from EKFalgorithm import ECM
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli  # check if poisson actually makes a difference or if bernoulli should be used
from bitstring import BitArray


def train_evaluate(ga_individual_sol):
    gene_length = 8
    R = ga_individual_sol[0:gene_length]
    P1 = ga_individual_sol[gene_length:2*gene_length]
    P2 = ga_individual_sol[2*gene_length:3*gene_length]
    P3 = ga_individual_sol[3*gene_length:4*gene_length]
    Q1 = ga_individual_sol[2*gene_length:3*gene_length]
    Q2 = ga_individual_sol[3*gene_length:4*gene_length]
    Q3 = ga_individual_sol[4*gene_length:5*gene_length]
    R = BitArray(R).uint
    P1 = BitArray(P1).uint
    P2 = BitArray(P2).uint
    P3 = BitArray(P3).uint
    Q1 = BitArray(Q1).uint
    Q2 = BitArray(Q2).uint
    Q3 = BitArray(Q3).uint

    # resize hyperparameterss to be within range
    R = np.interp(R, [0, 2**gene_length-1], [1e1, 1e3])
    P1 = np.interp(P1, [0, 2**gene_length-1], [1e-2, 1e2])
    P2 = np.interp(P2, [0, 2**gene_length-1], [1e-2, 1e2])
    P3 = np.interp(P3, [0, 2**gene_length-1], [1e-2, 1e2])
    Q1 = np.interp(Q1, [0, 2**gene_length-1], [1e-2, 1e2])
    Q2 = np.interp(Q2, [0, 2**gene_length-1], [1e-2, 1e2])
    Q3 = np.interp(Q3, [0, 2**gene_length-1], [1e-2, 1e2])

    print(f'R = {R}, P1 = {P1}, P2 = {P2}, P3 = {P3}, Q1 = {Q1}, Q2 = {Q2}, Q3 = {Q3}')

    # Run EKF
    try:
        ecm = ECM(R, P1, P2, P3, Q1, Q2, Q3, 'B0005')
        soc_est, vt_est, vt_err, total_err = ecm.EKF(with_discharge_cycles=True, save_plot=False)
    except np.linalg.LinAlgError:
        total_err = 1000000

    return [total_err]


if __name__ == '__main__':
    # init variables of ga using deap
    # set seed for reproducibility
    np.random.seed(121)
    # set number of generations
    ngen = 5
    # set population size
    popsize = 10
    # set gene length
    gene_length = 8
    entire_length = 7*gene_length

    # basically creates classes for the fitness and individual
    creator.create('FitnessMax', base.Fitness, weights=[-1.0])
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # create toolbox & initialize population (bernoulli random variables)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", bernoulli.rvs, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=entire_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ordered cross over for mating
    toolbox.register("mate", tools.cxOrdered)
    # mutation
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)
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
