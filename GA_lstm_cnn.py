from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from helpfunction import kfold_ind
import numpy as np
from bitstring import BitArray
from torch import cuda
import torch
import torch.backends.cudnn as cudnn
import random
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def basis_func(scaling_factor, hidden_layers):
    basis = np.linspace(1, scaling_factor, num=hidden_layers,  dtype=int)
    if hidden_layers == 1:
        basis[0] = 1
    # basis = (basis).astype(int)
    basis_fun = []
    basis_fun = []
    for i in range(hidden_layers):
        if basis[i] == 0:
            basis[i] = 1
        basis_fun.append(basis[i])
    return basis_fun


def train_evaluate(ga_individual_solution):
    cuda.empty_cache()
    gene_length = 8
    n_epoch = 100

    seq_steps = BitArray(ga_individual_solution[0:gene_length])
    num_layers_conv = BitArray(ga_individual_solution[gene_length:2*gene_length])
    output_channels = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
    kernel_sizes = BitArray(ga_individual_solution[3*gene_length:4*gene_length])
    stride_sizes = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
    padding_sizes = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
    hidden_size_lstm = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
    num_layers_lstm = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
    hidden_neurons_dense = BitArray(ga_individual_solution[8*gene_length:9*gene_length])
    lr = BitArray(ga_individual_solution[9*gene_length:10*gene_length])
    batch_size = BitArray(ga_individual_solution[10*gene_length:11*gene_length])

    seq_steps = seq_steps.uint
    num_layers_conv = num_layers_conv.uint
    output_channels = output_channels.uint
    kernel_sizes = kernel_sizes.uint
    stride_sizes = stride_sizes.uint
    padding_sizes = padding_sizes.uint
    hidden_size_lstm = hidden_size_lstm.uint
    num_layers_lstm = num_layers_lstm.uint
    hidden_neurons_dense = hidden_neurons_dense.uint
    lr = lr.uint
    batch_size = batch_size.uint

    # resize hyperparameterss to be within range
    seq_steps = int(np.interp(seq_steps, [0, 255], [50, 120]))
    num_layers_conv = int(np.interp(num_layers_conv, [0, 255], [1, 5]))
    output_channels = int(np.interp(output_channels, [0, 255], [1, 10]))
    kernel_sizes = int(np.interp(kernel_sizes, [0, 255], [1, 10]))
    stride_sizes = int(np.interp(stride_sizes, [0, 255], [1, 10]))
    padding_sizes = int(np.interp(padding_sizes, [0, 255], [1, 10]))
    hidden_size_lstm = int(np.interp(hidden_size_lstm, [0, 255], [1, 10]))
    num_layers_lstm = int(np.interp(num_layers_lstm, [0, 255], [1, 4]))
    hidden_neurons_dense = int(np.interp(hidden_neurons_dense, [0, 255], [1, 10]))
    lr = round(np.interp(lr, [0, 255], [0.0001, 0.1]), 5)
    batch_size = int(np.interp(batch_size, [0, 255], [150, 500]))

    output_channels = basis_func(output_channels, num_layers_conv)
    kernel_sizes = basis_func(kernel_sizes, num_layers_conv)
    stride_sizes = basis_func(stride_sizes, num_layers_conv)
    padding_sizes = basis_func(padding_sizes, num_layers_conv)
    hidden_neurons_dense = basis_func(hidden_neurons_dense, hidden_neurons_dense)
    hidden_neurons_dense_arr = np.flip(np.array(hidden_neurons_dense))
    hidden_neurons_dense = list(hidden_neurons_dense_arr)
    hidden_neurons_dense.append(1)
    hidden_neurons_dense[-1] = 1

    hyperparameters = [seq_steps, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]
    loss = kfold_ind(model_type='data', hyperparameters=hyperparameters, battery=['B0005', 'B0006', 'B0007', 'B0018'], plot=True, strict=True)
    return [loss]


torch.manual_seed(0)                       # Seed the RNG for all devices (both CPU and CUDA).
random.seed(0)                             # Set python seed for custom operators.
np.random.seed(0)             
torch.cuda.manual_seed_all(0) 
cudnn.deterministic = True
cudnn.benchmark = False

population_size = 50
num_generations = 25
entire_bit_array_length = 11*8

creator.create('FitnessMax', base.Fitness, weights=[-1.0])
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=entire_bit_array_length)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.8)
toolbox.register('select', tools.selTournament, tournsize=int(population_size/2))
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n=population_size)
r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.4, ngen=num_generations, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
print('Best ever individual = ', best_individual, '\nFitness = ', best_individual.fitness.values[0])
print(f'list of individuals = {best_individual}')

# data padded
# [1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 0, 2, 2, 2, 1, 2, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# 0.504700, seed 121
# Best ever individual =  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
# Fitness =  0.40241796018103054 seed 121

# hybrid padded
# Best ever individual =  [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
# Fitness =  0.45832835882902145 seed 121

# Best ever individual =  [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]
# Fitness =  0.3425200656056404 