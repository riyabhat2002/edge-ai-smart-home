#include <iostream>
#include <string>
#include <vector>
#include "llama.h"

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    int initialize(const std::string& model_path_str);
    int context_initialize();
    int vocab_initialize();
    int run_inference();
    llama_model* get_model() const { return m_model; }
    llama_context* get_context() const { return m_context; }
    const llama_vocab* get_vocab() const { return m_vocab; }

private:
    llama_model *m_model; 
    llama_context *m_context;
    const llama_vocab *m_vocab;
};