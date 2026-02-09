#pragma once

#include <string>
#include <winsock2.h>
#include <ws2tcpip.h>

class UdpReceiver {
public:
    UdpReceiver(int port);
    ~UdpReceiver();

    // Disable copy
    UdpReceiver(const UdpReceiver&) = delete;
    UdpReceiver& operator=(const UdpReceiver&) = delete;

    bool bind();
    void close();

    // Blocking receive with timeout for graceful shutdown
    // Returns empty string on timeout, data on receive
    std::string receive(int timeoutMs = 100);

    std::string getLastError() const;

private:
    int m_port;
    SOCKET m_socket;
    std::string m_lastError;
    bool m_wsaInitialized;

    bool initWinsock();
    void cleanupWinsock();
};
