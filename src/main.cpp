#include "InferenceEngine.h"
#include "MQTTClient.h"

int main(int argc, char *argv[])
{
    InferenceEngine engine;
    MQTTClient mqtt_client("edge_ai_client", "localhost", 1883);
    mosquitto* mqtt_client_ptr = mqtt_client.get_client();
    std::string model_path = "/edge-ai-smart-home/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    if (engine.initialize(model_path) != 0) {
        std::cerr << "Failed to initialize the inference engine." << std::endl;
        return 1;
    }

    if (engine.context_initialize() != 0) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    if (engine.vocab_initialize() != 0) {
        std::cerr << "Failed to initialize the vocabulary." << std::endl;
        return 1;
    }

    int err = engine.run_inference();
    if (err != 0) {
        std::cerr << "Inference failed." << std::endl;
        return 1;
    }

    std::cout << "Connecting to MQTT broker..." << std::endl;
    if (!mqtt_client.connect()) {
        std::cerr << "Failed to connect to MQTT broker." << std::endl;
        return 1;
    }

    std::cout << "Subscribing to MQTT topic..." << std::endl;

    if (!mqtt_client.subscribe("inference/requests")) {
        std::cerr << "Failed to subscribe to MQTT topic." << std::endl;
        return 1;
    }

    std::cout << "Publishing inference results to MQTT topic..." << std::endl;
    
    if (!mqtt_client.publish("inference/results", "Inference completed successfully.")) {
        std::cerr << "Failed to publish message to MQTT broker." << std::endl;
        return 1;
    }

    return 0;
}