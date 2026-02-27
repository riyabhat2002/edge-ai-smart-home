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
    
    llama_model_free(model);
    llama_backend_free();
    std::cout << "Llama backend freed successfully." << std::endl;

    return 0;
}