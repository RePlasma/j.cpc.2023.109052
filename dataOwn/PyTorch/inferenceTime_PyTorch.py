# PyTorch inference time
#
# function getModel(d,L) gives average inference time on a model with L hidden layers and d neurons each
# inference is repeated Nrepeat times for average
# inference is run on batchSize = 1000
#
# processor: 3,5 GHz 6-Core Intel Xeon E5 
#
# Óscar Amaro (Jan 2024)
#

import torch
import torch.nn as nn
from tqdm import trange
import timeit
import numpy as np
import time
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt

def getModelInferenceTime(L,d):
    input_size = 2
    num_layers = L 
    num_neurons = d
    output_size = 1
    batchSize = 1000
    Nrepeats = 300
    
    # build model NN
    class CustomMLP(nn.Module):
        def __init__(self):
            super(CustomMLP, self).__init__()
            layers = []

            # Input layer
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.ReLU())

            # Hidden layers
            for i in range(num_layers):
                layers.append(nn.Linear(num_neurons, num_neurons))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(num_neurons, output_size))
            layers.append(nn.Sigmoid())

            # Combine all layers
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    # Create the PyTorch model
    pytorch_model = CustomMLP()

    # Time the PyTorch model
    start = time.time()
    for i in range(Nrepeats):
        pytorch_model(torch.rand(batchSize,input_size))
    end = time.time()
    
    # return average repeated inference time
    return (end - start)/Nrepeats

# arrays of L layers and d neurons per layer
Llst = np.array([1,5,10,25,50])
dlst = np.array([10,25,50,100])

# timings arrays
timelst1 = np.zeros((len(Llst),len(dlst)))
timelst2 = np.zeros((len(Llst),len(dlst)))
# first run
for i in trange(len(Llst)):
    for j in range(len(dlst)):
        timelst1[i,j] = getModelInferenceTime(Llst[i], dlst[j])
# second run
for i in trange(len(Llst)):
    for j in range(len(dlst)):
        timelst2[i,j] = getModelInferenceTime(Llst[i], dlst[j])
# average of timings
timelst = (timelst1 + timelst2)/2

pd.DataFrame(timelst).to_csv('inferenceTime_PyTorch.csv', index=False)
plt.plot(Llst, timelst)
plt.xlabel(r'L')
plt.ylabel(r'time [s]')
plt.show()