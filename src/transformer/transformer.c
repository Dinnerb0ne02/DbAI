#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <network/layer.h>
#include <transformer/transformer.h>

// Helper function for residual connection
static void apply_residual_connection(float* target, const float* source, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        target[i] += source[i];
    }
}

Transformer* transformer_init(size_t num_layers, size_t d_model, size_t num_heads, size_t dff) {
    Transformer* transformer = malloc(sizeof(Transformer));
    if (!transformer) return NULL;

    transformer->num_layers = num_layers;
    transformer->d_model = d_model;
    transformer->num_heads = num_heads;
    transformer->dff = dff;

    transformer->blocks = malloc(num_layers * sizeof(TransformerBlock));
    if (!transformer->blocks) {
        free(transformer);
        return NULL;
    }

    for (size_t i = 0; i < num_layers; ++i) {
        // Initialize self-attention layer
        size_t qkv_size = d_model / num_heads * 3; // Q, K, V
        transformer->blocks[i].self_attention = create_layer(d_model, qkv_size);
        if (!transformer->blocks[i].self_attention) {
            transformer_free(transformer);
            return NULL;
        }

        // Initialize feed-forward layer
        transformer->blocks[i].feed_forward = create_layer(d_model, dff);
        if (!transformer->blocks[i].feed_forward) {
            transformer_free(transformer);
            return NULL;
        }
    }

    return transformer;
}


void transformer_forward(Transformer* transformer, const float* input, float* output) {
    // Validate inputs
    if (!transformer || !input || !output) return;

    // Copy input to output as starting point
    memcpy(output, input, transformer->d_model * sizeof(float));

    // Allocate temporary buffers
    float* self_attention_output = malloc(transformer->d_model * sizeof(float));
    float* feed_forward_output = malloc(transformer->d_model * sizeof(float));
    if (!self_attention_output || !feed_forward_output) {
        free(self_attention_output);
        free(feed_forward_output);
        return;
    }

    for (size_t i = 0; i < transformer->num_layers; ++i) {
        // 1. Self-attention layer
        layer_forward(transformer->blocks[i].self_attention, output, self_attention_output);

        // 2. Residual connection (self-attention output + original input)
        apply_residual_connection(self_attention_output, output, transformer->d_model);

        // 3. Feed-forward layer
        layer_forward(transformer->blocks[i].feed_forward, self_attention_output, feed_forward_output);

        // 4. Residual connection (feed-forward output + self-attention output)
        apply_residual_connection(feed_forward_output, self_attention_output, transformer->d_model);

        // 5. Prepare for next layer
        memcpy(output, feed_forward_output, transformer->d_model * sizeof(float));
    }

    // Cleanup
    free(self_attention_output);
    free(feed_forward_output);
}

void transformer_free(Transformer* transformer) {
    if (!transformer) return;

    if (transformer->blocks) {
        for (size_t i = 0; i < transformer->num_layers; ++i) {
            if (transformer->blocks[i].self_attention) {
                free_layer(transformer->blocks[i].self_attention);
            }
            if (transformer->blocks[i].feed_forward) {
                free_layer(transformer->blocks[i].feed_forward);
            }
        }
        free(transformer->blocks);
    }
    free(transformer);
}