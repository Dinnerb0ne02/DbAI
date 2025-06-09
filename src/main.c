#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <network/layer.h>
#include <network/network.h>

#include <activation/relu.h>
#include <activation/sigmoid.h>

#include <optimizer/adam.h>
#include <optimizer/sgd.h>

#include <data/data_loader.h>

#include <loss/cross_entropy.h>
#include <loss/mse.h>

#include <utils/utils.h> 

#define NUM_SAMPLES 1000
#define INPUT_SIZE 784  // e.g., 28x28 pixels for MNIST
#define HIDDEN_LAYER_SIZE 128
#define NUM_CLASSES 10  // e.g., number of classes for MNIST
#define EPOCHS 10
#define LEARNING_RATE 0.001
#define BATCH_SIZE 32

int main() {
    // Initialize network
    Network* net = create_network(3);
    if (!net) {
        fprintf(stderr, "Failed to create network.\n");
        return EXIT_FAILURE;
    }
    if (add_layer(net, INPUT_SIZE, HIDDEN_LAYER_SIZE) != 0) {
        fprintf(stderr, "Failed to add hidden layer.\n");
        free_network(net);
        return EXIT_FAILURE;
    }
    if (add_layer(net, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) != 0) {
        fprintf(stderr, "Failed to add second hidden layer.\n");
        free_network(net);
        return EXIT_FAILURE;
    }
    if (add_layer(net, HIDDEN_LAYER_SIZE, NUM_CLASSES) != 0) {
        fprintf(stderr, "Failed to add output layer.\n");
        free_network(net);
        return EXIT_FAILURE;
    }

    // Initialize weights
    for (size_t i = 0; i < net->num_layers; ++i) {
        initialize_weights(net->layers[i]->weights, net->layers[i]->input_size * net->layers[i]->output_size);
    }

    // Load data
    DataLoader* train_loader = create_data_loader("data/train_data.txt", NUM_SAMPLES, INPUT_SIZE);
    if (!train_loader) {
        fprintf(stderr, "Failed to load training data.\n");
        free_network(net);
        return EXIT_FAILURE;
    }

    // Create Adam optimizer
    Adam* optimizer = adam_init(net->layers[net->num_layers - 1]->output_size, LEARNING_RATE, 0.9, 0.999, 1e-8);
    if (!optimizer) {
        fprintf(stderr, "Failed to create Adam optimizer.\n");
        free_data_loader(train_loader);
        free_network(net);
        return EXIT_FAILURE;
    }

    // Training loop
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        printf("Epoch %zu/%zu\n", epoch + 1, EPOCHS);
        for (size_t batch_index = 0; batch_index < NUM_SAMPLES / BATCH_SIZE; ++batch_index) {
            float* batch_inputs;
            float* batch_labels;
            if (!get_batch(train_loader, batch_index, &batch_inputs, &batch_labels)) {
                fprintf(stderr, "Failed to get batch data.\n");
                continue;
            }

            // Forward pass
            float* outputs = malloc(net->layers[net->num_layers - 1]->output_size * BATCH_SIZE * sizeof(float));
            forward_pass_network(net, batch_inputs, outputs);

            // Compute loss
            float loss = cross_entropy(outputs, batch_labels, net->layers[net->num_layers - 1]->output_size * BATCH_SIZE);

            // Backward pass
            backward_pass_network(net, batch_labels, LEARNING_RATE);

            // Update weights
            for (size_t i = 0; i < net->num_layers; ++i) {
                adam_update(optimizer, net->layers[i]->weights, net->layers[i]->gradients, net->layers[i]->input_size * net->layers[i]->output_size);
            }

            // Cleanup
            free(outputs);

            printf("Batch %zu/%zu, Loss: %f\n", batch_index + 1, NUM_SAMPLES / BATCH_SIZE, loss);
        }
    }

    // Evaluate model performance
    DataLoader* test_loader = create_data_loader("data/test_data.txt", NUM_SAMPLES, INPUT_SIZE);
    if (!test_loader) {
        fprintf(stderr, "Failed to load test data.\n");
        free_network(net);
        adam_free(optimizer);
        return EXIT_FAILURE;
    }
    float accuracy = evaluate_network(net, test_loader);
    printf("Test Accuracy: %f%%\n", accuracy * 100);

    // Cleanup
    free_data_loader(train_loader);
    free_data_loader(test_loader);
    free_network(net);
    adam_free(optimizer);

    return EXIT_SUCCESS;
}