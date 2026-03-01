#include "InferenceEngine.h"

int main(int argc, char *argv[])
{
    InferenceEngine engine;
    std::string model_path = "/edge-ai-smart-home/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    if (engine.initialize(model_path) != 0) {
        std::cerr << "Failed to initialize the inference engine." << std::endl;
        return 1;
    }

    llama_model *model = engine.get_model();
    if (model == nullptr) {
        std::cerr << "Model is not loaded properly." << std::endl;
        return 1;
    }

    if (engine.context_initialize() != 0) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }
    llama_context *context = engine.get_context();
    if (context == nullptr) {
        std::cerr << "Context is not initialized properly." << std::endl;
        return 1;
    }

    if (engine.vocab_initialize() != 0) {
        std::cerr << "Failed to initialize the vocabulary." << std::endl;
        return 1;
    }
    const llama_vocab *vocab = engine.get_vocab();
    if (vocab == nullptr) {
        std::cerr << "Vocabulary is not initialized properly." << std::endl;
        return 1;
    }
    int err = engine.run_inference();
    if (err != 0) {
        std::cerr << "Inference failed." << std::endl;
        return 1;
    }

    return 0;
}