#include <iostream>
#include <string>
#include "llama.h"

int main()
{
    llama_backend_init();
    std::cout << "Llama backend initialized successfully." << std::endl;

    std::string home = std::getenv("HOME");
    std::string model_path = home + "/edge-ai-smart-home/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // Load all layers into VRAM (if possible)
    model_params.use_mmap = false; // Disable memory mapping for better performance on small models
    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == nullptr) {
        std::cerr << "Failed to load the model from " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully from " << model_path << std::endl;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512; // Set context size to 512 tokens
    context_params.n_threads = 3; // Use 3 threads for generation
    context_params.type_k = GGML_TYPE_Q4_0; // Use 4-bit quantization for K cache
    context_params.type_v = GGML_TYPE_Q4_0; // Use 4-bit quantization for V cache

    llama_context * context = llama_init_from_model(model, context_params);
    
    if (context == nullptr) {
        std::cerr << "Failed to initialize the context from the model." << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::cout << "Context initialized successfully." << std::endl;

    llama_model_free(model);
    llama_backend_free();
    std::cout << "Llama backend freed successfully." << std::endl;

    return 0;
}