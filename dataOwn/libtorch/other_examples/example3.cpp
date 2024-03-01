// libtorch (torch c++): time forward pass on network with random w&b
// averaging inference time over nrepeat times
// Óscar Amaro (Feb 2024)

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

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

int main() {

    // Define the number of neurons in each layer.
    std::vector<int> layers_sizes = {8, 8, 8, 1};

    // Create the network.
    Net net(layers_sizes);

    // Initialize weights and biases.
    for(auto& layer : net.layers) {
        layer->weight.data().uniform_(-0.1, 0.1);
        layer->bias.data().uniform_(-0.1, 0.1);
    }

    // Create a random input tensor.
    auto input = torch::randn({1, layers_sizes[0]});
    std::cout << "Input: " << input << std::endl;


    // Measure the forward pass time.
    int nrepeat = 3;  // Number of times to repeat the forward pass.
    std::chrono::microseconds total_duration(0);
    auto output = net.forward(input);
    for (int i = 0; i < nrepeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = net.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    // Calculate and print the duration.
    std::cout << "Output: " << output << std::endl;
    std::cout << "Average forward pass time over " << nrepeat << " runs: " 
              << total_duration.count() / static_cast<double>(nrepeat) << " microseconds." << std::endl;


    return 0;
}
