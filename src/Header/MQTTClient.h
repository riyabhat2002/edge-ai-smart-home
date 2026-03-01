#include "mosquitto.h"
#include <string>
#include <condition_variable>
#include <mutex>
#include <atomic>

class MQTTClient {
public:
    MQTTClient(const std::string& client_id, const std::string& host, int port);
    ~MQTTClient();
    bool connect();
    bool subscribe(const std::string& topic);
    bool publish(const std::string& topic, const std::string& message);
    bool disconnect();
    mosquitto* get_client() const { return m_mosq; }

private:
    static void on_publish_callback(struct mosquitto *mosq, void *obj, int mid);
    struct mosquitto* m_mosq;
    std::string client_id;
    std::string host;
    int port;
    std::mutex m_publish_mtx;
    std::condition_variable m_publish_cv;
    std::atomic<bool> m_publish_acknowledged{false};
    int m_last_mid{0};
};