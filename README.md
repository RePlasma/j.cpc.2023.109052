# RoseNNa: A performant, portable library for neural network inference with application to computational fluid dynamics

Original authors of the paper: Ajay Bati, Spencer H. Bryngelson

Link to paper: https://www.sciencedirect.com/science/article/pii/S0010465523003971?via%3Dihub

Link to pre-print: https://arxiv.org/abs/2307.16322

Link to github repo: https://github.com/comp-physics/roseNNa (last commit Aug 27, 2023)

Author of this notebook: [Óscar Amaro](https://github.com/OsAmaro) (December 2023)

To install RoseNNa and check that it is working ok, check this [guide](https://github.com/RePlasma/j.cpc.2023.109052/blob/main/guide_RoseNNa/guide_RoseNNa.md).

Data from preprint retrieved with WebPlotDigitizer. Results of notebook that are not from the paper were obtained with a ``3.5 GHz 6-Core Intel Xeon E5`` processor.

Outline:
- [Figure 2](#Figure2): Multilayer perceptron (MLP) time comparison (**RoseNNa vs PyTorch**)
- [Figure 3](#Figure3): Long Short-Term Memory (LSTM) time comparison (**RoseNNa vs PyTorch**)
- [Figure 4](#Figure4): Multilayer perceptron (MLP) model time comparison (**RoseNNa vs libtorch**)

Abstract of paper: _The rise of neural network-based machine learning ushered in high-level libraries, including TensorFlow and PyTorch, to support their functionality. Computational fluid dynamics (CFD) researchers have benefited from this trend and produced powerful neural networks that promise shorter simulation times. For example, multilayer perceptrons (MLPs) and Long Short Term Memory (LSTM) recurrent-based (RNN) architectures can represent sub-grid physical effects, like turbulence. Implementing neural networks in CFD solvers is challenging because the programming languages used for machine learning and CFD are mostly non-overlapping, We present the roseNNa library, which bridges the gap between neural network inference and CFD. RoseNNa is a non-invasive, lightweight (1000 lines), and performant tool for neural network inference, with focus on the smaller networks used to augment PDE solvers, like those of CFD, which are typically written in C/C++ or Fortran. RoseNNa accomplishes this by automatically converting trained models from typical neural network training packages into a high-performance Fortran library with C and Fortran APIs. This reduces the effort needed to access trained neural networks and maintains performance in the PDE solvers that CFD researchers build and rely upon. Results show that RoseNNa reliably outperforms PyTorch (Python) and libtorch (C++) on MLPs and LSTM RNNs with less than 100 hidden layers and 100 neurons per layer, even after removing the overhead cost of API calls. Speedups range from a factor of about 10 and 2 faster than these established libraries for the smaller and larger ends of the neural network size ranges tested._
