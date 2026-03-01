#include "MQTTClient.h"
#include <stdexcept>
#include <iostream>

MQTTClient::MQTTClient(const std::string& client_id, const std::string& host, int port)
    : client_id(client_id), host(host), port(port) {
    mosquitto_lib_init();
    m_mosq = mosquitto_new(client_id.c_str(), true, this);
    if (!m_mosq) {
        throw std::runtime_error("Failed to create MQTT client.");
    }
    mosquitto_publish_callback_set(m_mosq, MQTTClient::on_publish_callback);
}

MQTTClient::~MQTTClient() {
    if(m_mosq) {
        mosquitto_loop_stop(m_mosq, true);
        disconnect();
        mosquitto_destroy(m_mosq);
    }
    mosquitto_lib_cleanup();
}

bool MQTTClient::connect() {
    int ret = mosquitto_connect(m_mosq, host.c_str(), port, 60);
    if(ret != MOSQ_ERR_SUCCESS) {
        std::cout << "Failed to connect to MQTT broker: " << mosquitto_strerror(ret) << std::endl;
        return false;
    }   

    mosquitto_loop_start(m_mosq); 
    return ret == MOSQ_ERR_SUCCESS;
}

bool MQTTClient::subscribe(const std::string& topic) {
    int ret = mosquitto_subscribe(m_mosq, nullptr, topic.c_str(), 1);
    return ret == MOSQ_ERR_SUCCESS;
}

bool MQTTClient::publish(const std::string& topic, const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(m_publish_mtx);
        m_publish_acknowledged = false;
    }
    int mid;
    int ret = mosquitto_publish(m_mosq, &mid, topic.c_str(), message.size(), message.c_str(), 1, false);
    if(ret != MOSQ_ERR_SUCCESS) {
        std::cout << "Failed to publish message: " << mosquitto_strerror(ret) << std::endl;
        return false;
    }
    
    std::unique_lock<std::mutex> lock(m_publish_mtx);
    bool signaled = m_publish_cv.wait_for(lock, std::chrono::seconds(5), [this]() {
        return m_publish_acknowledged.load();
    });

    if (!signaled) {
        std::cerr << "Publish timed out after 5 seconds." << std::endl;
        return false;
    }

    std::cout << "Publish confirmed by broker (mid: " << mid << ")" << std::endl;
    return ret == MOSQ_ERR_SUCCESS;
}

void MQTTClient::on_publish_callback(struct mosquitto *mosq, void *obj, int mid) {
    auto* self = static_cast<MQTTClient*>(obj);
    
    std::lock_guard<std::mutex> lock(self->m_publish_mtx);
    self->m_publish_acknowledged = true;
    self->m_publish_cv.notify_all(); // Wake up the publish() function
}

bool MQTTClient::disconnect() {
    int ret = mosquitto_disconnect(m_mosq);
    return ret == MOSQ_ERR_SUCCESS;
}

