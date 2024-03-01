# roseNNa MLP with ReLU activations, L=30 d=30
# export model to onnx for conversion by roseNNa
# for inference time with varying batch_size
#
# Óscar Amaro (Mar 2024)

import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pathlib
import sys, getopt

opts, args = getopt.getopt(sys.argv[1:],"n")
produce = True
for opt, _ in opts:
    if opt == "-n":
        produce = False
        
# build model NN
input_size = 2
num_layers = 30
num_neurons = 30
output_size = 1
#batchSize = 1000
#Nrepeats = 300
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
        layers.append(nn.ReLU())

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


model = CustomMLP()

inp = torch.ones(1,2)
if produce:
    with open("inputs.fpp",'w') as f:
        inputs = inp.flatten().tolist()
        inpShapeDict = {'inputs': list(inp.shape)}
        inpDict = {'inputs':inputs}
        f.write(f"""#:set inpShape = {inpShapeDict}""")
        f.write("\n")
        f.write(f"""#:set arrs = {inpDict}""")
        f.write("\n")
        f.write("a")

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

logits = model(inp)
filePath = "../my_roseNNa_examples/L30d30/"
with open(filePath+"L30d30.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"L30d30.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"L30d30_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
