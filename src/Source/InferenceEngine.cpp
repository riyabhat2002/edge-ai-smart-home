#include "InferenceEngine.h"

InferenceEngine::InferenceEngine(){}

InferenceEngine::~InferenceEngine()
{
    if (m_model) {
        llama_model_free(m_model);
    }
    llama_backend_free();
    std::cout << "Llama backend freed successfully." << std::endl;
}

int InferenceEngine::initialize(const std::string& model_path_str) {
    int err = 0;
    llama_backend_init();
    std::cout << "Llama backend initialized successfully." << std::endl;
    std::string home = std::getenv("HOME");
    std::string model_path = home + model_path_str;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // There is no GPU available, so set to 0 to load all layers into RAM
    model_params.use_mmap = false; // Disable memory mapping for better performance on small models
    m_model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (m_model == nullptr) {
        std::cerr << "Failed to load the model from " << model_path << std::endl;
        err = 1;;
    }
    std::cout << "Model loaded successfully from " << model_path << std::endl;
    return err;
}

int InferenceEngine::context_initialize() {
    int err = 0;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512; // Set context size to 512 tokens
    context_params.n_threads = 3; // Use 3 threads for generation
    context_params.type_k = GGML_TYPE_Q8_0; // Use 8-bit quantization for K cache
    context_params.type_v = GGML_TYPE_Q8_0;
    

    m_context = llama_init_from_model(m_model, context_params);
    
    if (m_context == nullptr) {
        std::cerr << "Failed to initialize the context from the model." << std::endl;
        llama_model_free(m_model);
        llama_backend_free();
        err = 1;;
    }
    std::cout << "Context initialized successfully." << std::endl;
    return err;
}

int InferenceEngine::vocab_initialize() {
    int err = 0;
    m_vocab = llama_model_get_vocab(m_model);

    if (m_vocab == nullptr) {
        std::cerr << "Failed to retrieve the vocabulary from the model." << std::endl;
        return 1;
    }

    std::cout << "Vocabulary retrieved successfully." << std::endl;
    return err;
}

int InferenceEngine::run_inference() {
    int err = 0;
    std::cin >> std::ws; // Clear any leading whitespace from the input stream
    std::string input_text;
    // Wait for user input before proceeding with tokenization
    std::getline(std::cin, input_text);

    std::string prompt = "<|start_header_id|>system<|end_header_id|>\n"
    "You are a smart home controller. You must ONLY respond with a JSON object "
    "containing device, action, and parameters fields. Never respond with plain text.\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    + input_text +
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    int32_t n_tokens_max = 128;
    std::vector<llama_token> token_buffer(n_tokens_max); // Create a buffer to hold the tokens
    int32_t text_len = prompt.length(); // Length of the input text "It is dark in here."
    if(text_len == 0) {
        std::cerr << "Input text is empty. Please provide a valid input." << std::endl;   
        err = 1;
    }
    int32_t tokens_count = llama_tokenize(m_vocab, prompt.c_str(), text_len, token_buffer.data(), token_buffer.size(),
                            true,
                            true); 

    if (tokens_count < 0) {
        std::cerr << "Failed to tokenize the input text." << std::endl;  
        err = 1;
    } else {
        std::cout << "Tokenization successful. Number of tokens: " << tokens_count << std::endl;
        std::cout << "Tokens: ";
        for (int32_t i = 0; i < tokens_count; ++i) {
            std::cout << token_buffer[i] << " ";  
        }
        std::cout << std::endl;
    }

    llama_batch batch = llama_batch_get_one(token_buffer.data(), tokens_count);

    int32_t err_code = llama_decode(m_context, batch);

    if (err_code != 0) {
        std::cerr << "Failed to decode the batch of tokens. Error code: " << err_code << std::endl;
        err = 1;
    } else {
        std::cout << "Batch decoded successfully." << std::endl;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false; // Enable performance timings for the sampler chain
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string output_text;
    int max_token_out = 100; // Maximum number of tokens to sample

    /* inference loop */
    for(int i = 0; i < max_token_out; ++i) {
        llama_token next_token;
        next_token = llama_sampler_sample(sampler, m_context, -1); // Sample from the last token in the batch

        if(next_token < 0) {
            std::cerr << "Failed to sample the next token. Error code: " << next_token << std::endl;
            err = 1;
            break;
        }   

        if (llama_vocab_is_eog(m_vocab, next_token)) { // Assuming 0 is the end-of-sequence token
            break;
        }
        
        char buffer[256];
        int32_t piece_len = llama_token_to_piece(m_vocab, next_token, buffer, sizeof(buffer), 0, false);
        if (piece_len > 0) {
            output_text += std::string(buffer, piece_len);
        }

        batch = llama_batch_get_one(&next_token, 1);
        err_code = llama_decode(m_context, batch);
        if(err_code != 0) {
            std::cerr << "Failed to decode the next token. Error code: " << err_code << std::endl;
            err = 1;
            break;
        }
    }

    std::cout << "Detokenized output: " << output_text << std::endl;
    return err;
}