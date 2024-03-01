// libtorch (torch c++): time forward pass for L hidden layers and delta neurons each, batchSize = 1000
// Óscar Amaro (Feb 2024)

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;

    Net(const std::vector<int>& layers_sizes) {
        for(size_t i = 0; i < layers_sizes.size() - 1; ++i) {
            layers.push_back(register_module("fc" + std::to_string(i), torch::nn::Linear(layers_sizes[i], layers_sizes[i + 1])));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        for(size_t i = 0; i < layers.size() - 1; ++i) {
            x = torch::relu(layers[i]->forward(x));
        }
        x = layers.back()->forward(x);
        return x;
    }
};

double getModelInferenceTime(int L, int d) {

    int batchSize = 1000; // batch size for inference
    int Nrepeats = 300;  // Number of times to repeat the forward pass.

    // model will have L hidden layers
    std::vector<int> layers_sizes(L+2, d);
    // Adjusting first and last layer sizes
    layers_sizes[0] = 2;/* appropriate input layer size */;
    layers_sizes.back() = 1;/* appropriate output layer size */;

    // Create the network.
    Net net(layers_sizes);

    // Initialize weights and biases.
    for(auto& layer : net.layers) {
        layer->weight.data().uniform_(-0.1, 0.1);
        layer->bias.data().zero_();
    }

    // Create a random input tensor with batch size.
    auto input = torch::randn({batchSize, layers_sizes[0]});
    //std::cout << "Input: " << input << std::endl;

    // Measure the forward pass time, Nrepeats times
    std::chrono::microseconds total_duration(0);
    auto output = net.forward(input);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Nrepeats; ++i) {
        output = net.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Calculate output and print the duration.
    //std::cout << "Output: " << output << std::endl;
    //std::cout << "Average forward pass time over " << Nrepeats << " runs: " << total_duration.count() / static_cast<double>(Nrepeats) << " microseconds." << std::endl;

    return total_duration.count() / static_cast<double>(Nrepeats) / 1.0e6;
}

int main() {


    // arrays of L layers and d neurons per layer
    std::vector<int> Llst = {1,5,10,25,50};
    std::vector<int> dlst = {10,25,50,100};

    // Dynamically allocate memory for 2D timings arrays
    double** timelst = new double*[Llst.size()];
    for (int i = 0; i < Llst.size(); ++i) {
        timelst[i] = new double[dlst.size()];
    }

    // timing
    std::cout << "(L,d) libtorch inference time [s] \n";
    for (int i = 0; i < Llst.size(); ++i) {
        for (int j = 0; j < dlst.size(); ++j) {
            timelst[i][j] = getModelInferenceTime(Llst[i], dlst[j]);
            std::cout << "(" << Llst[i] << "," << dlst[j] << ") " << timelst[i][j] << "\n";
        }
    }

    // save results to .csv
    // Opening a file in write mode.
    std::ofstream file("../inferenceTime_libtorch.csv");
    if (file.is_open()) {
        // Looping through the 2D array
        for (int i = 0; i < Llst.size(); ++i) {
            for (int j = 0; j < dlst.size(); ++j) {
                file << timelst[i][j] << ","; // Comma for separating elements
            }
            file << "\n"; // New line at the end of each row
        } 
        file.close(); // Closing the file
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
