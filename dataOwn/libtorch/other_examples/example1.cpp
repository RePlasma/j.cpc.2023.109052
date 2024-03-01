// libtorch (torch c++): simple forward pass on network with random w&b
// Óscar Amaro (Feb 2024)
#include <torch/torch.h>
#include <iostream>

// Define a new Module.
struct Net : torch::nn::Module {
    Net() {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(8, 8));
        fc2 = register_module("fc2", torch::nn::Linear(8, 8));
        fc3 = register_module("fc3", torch::nn::Linear(8, 1));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Use torch::relu to apply the Rectified Linear Unit (ReLU) function.
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        // No activation after the final layer.
        x = fc3->forward(x);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
    // Create a new Net.
    Net net;

    // Initialize weights and biases for each layer.
    net.fc1->weight.data().uniform_(-0.1, 0.1);
    net.fc1->bias.data().uniform_(-0.1, 0.1);
    net.fc2->weight.data().uniform_(-0.1, 0.1);
    net.fc2->bias.data().uniform_(-0.1, 0.1);
    net.fc3->weight.data().uniform_(-0.1, 0.1);
    net.fc3->bias.data().uniform_(-0.1, 0.1);

    // Create a random input tensor.
    auto input = torch::randn({1, 8});

    // Perform forward pass and print the output.
    std::cout << "Input: " << input << std::endl;
    auto output = net.forward(input);
    std::cout << "Output: " << output << std::endl;

    return 0;
}
