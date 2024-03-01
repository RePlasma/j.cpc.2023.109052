// libtorch (torch c++): time forward pass , batch inference
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
    // Define the number of neurons in each layer and batch size.
    std::vector<int> layers_sizes = {8, 8, 8, 1};
    int batch_size = 5;

    // Create the network.
    Net net(layers_sizes);

    // Initialize weights and biases.
    for(auto& layer : net.layers) {
        layer->weight.data().uniform_(-0.1, 0.1);
        layer->bias.data().uniform_(-0.1, 0.1);
    }

    // Create a random input tensor with batch size.
    auto input = torch::randn({batch_size, layers_sizes[0]});
    std::cout << "Input: " << input << std::endl;

    // Measure the forward pass time.
    auto start = std::chrono::high_resolution_clock::now();
    auto output = net.forward(input);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Output: " << output << std::endl;

    // Calculate and print the duration.
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Forward pass took " << duration.count() << " microseconds." << std::endl;

    return 0;
}
