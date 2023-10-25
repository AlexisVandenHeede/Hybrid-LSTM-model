from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from helpfunction import kfold_ind
import numpy as np
from bitstring import BitArray
from torch import cuda
import torch


def basis_func(scaling_factor, hidden_layers):
    basis = np.linspace(3, scaling_factor, num=hidden_layers,  dtype=int)
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

    seq_length = BitArray(ga_individual_solution[0:gene_length])
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

    seq_length = seq_length.uint
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
    seq_length = int(np.interp(seq_length, [0, 255], [1, 100]))
    num_layers_conv = int(np.interp(num_layers_conv, [0, 255], [3, 4]))
    k1 = int(np.interp(output_channels, [0, 255], [1, 50]))
    kernel_sizes = int(np.interp(kernel_sizes, [0, 255], [3, 11]))
    stride_sizes = int(np.interp(stride_sizes, [0, 255], [2, 15]))
    padding_sizes = int(np.interp(padding_sizes, [0, 255], [1, 10]))
    hidden_size_lstm = int(np.interp(hidden_size_lstm, [0, 255], [3, 20]))
    num_layers_lstm = int(np.interp(num_layers_lstm, [0, 255], [1, 5]))
    hidden_neurons_dense = int(np.interp(hidden_neurons_dense, [0, 255], [0.0001, 0.5]))
    lr = round(np.interp(lr, [0, 255], [0.0001, 0.1]), 5)
    batch_size = int(np.interp(batch_size, [0, 255], [150, 2000]))

    k2 = hidden_neurons_dense
    k3 = stride_sizes
    kernel_sizes = basis_func(kernel_sizes, num_layers_conv)
    stride_sizes = basis_func(stride_sizes, num_layers_conv)
    padding_sizes = basis_func(padding_sizes, num_layers_conv)
 
 
    
    num_layers_conv = 1
    stride_sizes = [1]*num_layers_conv
    padding_sizes= [1]*num_layers_conv
  
    num_layers_lstm = 1
    
    output_channels = [k1]#, k2, k3, 1
    kernel_sizes = [3]#*num_layers_conv
        
    #to optimise: seq, num_layers_conv, output_channels (num of kernels), hidden_size_lstm, num_layers_lstm, lr, batch_size, n_epoch
    hyperparameters = [seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]
    #[26, 4, [13, 18, 7, 1], [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 4, 1, None, 0.04515, 1833, 100]
    print(f'hyperparameters: {hyperparameters}')
    loss = kfold_ind(model_type='data_padded', hyperparameters=hyperparameters, battery=['B0005', 'B0006', 'B0007', 'B0018'], plot=False, strict=True)
    return [loss]

def train_evaluate_new(ga_individual_solution):
    cuda.empty_cache()
    gene_length = 8
    n_epoch = 100

    num_kernels = BitArray(ga_individual_solution[0:gene_length])
    kernel_sizes = BitArray(ga_individual_solution[gene_length:2*gene_length])
    stride_sizes = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
    padding_sizes = BitArray(ga_individual_solution[3*gene_length:4*gene_length])
    hidden_size_lstm = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
    num_layers_lstm = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
    slope = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
    seq_length = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
    lr = BitArray(ga_individual_solution[8*gene_length:9*gene_length])
    batch_size = BitArray(ga_individual_solution[9*gene_length:10*gene_length])
    dense_neurons = BitArray(ga_individual_solution[10*gene_length:11*gene_length])


    num_kernels = num_kernels.uint
    kernel_sizes = kernel_sizes.uint
    hidden_size_lstm = hidden_size_lstm.uint
    num_layers_lstm = num_layers_lstm.uint
    slope = slope.uint
    seq_length = seq_length.uint
    lr = lr.uint
    batch_size = batch_size.uint
    dense_neurons = dense_neurons.uint

    # resize hyperparameterss to be within range
    num_kernels = int(np.interp(num_kernels, [0, 255], [1, 100]))
    kernel_sizes = int(np.interp(kernel_sizes, [0, 255], [1, 10]))
    hidden_size_lstm = int(np.interp(hidden_size_lstm, [0, 255], [1, 30]))
    num_layers_lstm = int(np.interp(num_layers_lstm, [0, 255], [1, 5]))
    slope = int(np.interp(slope, [0, 255], [1, 5000]))
    seq_length = int(np.interp(seq_length, [0, 255], [1, 100]))
    lr = round(np.interp(lr, [0, 255], [0.0001, 0.1]), 5)
    batch_size = int(np.interp(batch_size, [0, 255], [150, 2000]))
    dense_neurons = int(np.interp(dense_neurons, [0, 255], [1, 100]))

    kernel_sizes = [3, 3]
    stride_sizes = [1, 1]
    padding_sizes = [1, 1]
    slope = slope/1000
  
    #to optimise: seq, num_layers_conv, output_channels (num of kernels), hidden_size_lstm, num_layers_lstm, lr, batch_size, n_epoch
    hyperparameters = [num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, slope, seq_length, 4, lr, batch_size, dense_neurons]
    #[26, 4, [13, 18, 7, 1], [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 4, 1, None, 0.04515, 1833, 100]
    print(f'hyperparameters: {hyperparameters}')
    loss = kfold_ind(model_type='data_padded', hyperparameters=hyperparameters, battery=['B0005', 'B0006', 'B0007', 'B0018'], plot=False, strict=True)
    return [loss]


population_size = 50
num_generations = 5
entire_bit_array_length = 11*8

creator.create('FitnessMax', base.Fitness, weights=[-1.0])
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=entire_bit_array_length)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
toolbox.register('select', tools.selTournament, tournsize=int(population_size/2))
toolbox.register('evaluate', train_evaluate_new)

population = toolbox.population(n=population_size)
r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
print('Best ever individual = ', best_individual, '\nFitness = ', best_individual.fitness.values[0])
print(f'list of individuals = {best_individual}')

