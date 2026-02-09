#include "udp_receiver.hpp"
#include <iostream>
#include <cstring>

// Link with Ws2_32.lib (MSVC only)
#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif

constexpr size_t UDP_BUFFER_SIZE = 256;

UdpReceiver::UdpReceiver(int port)
    : m_port(port)
    , m_socket(INVALID_SOCKET)
    , m_wsaInitialized(false)
{
}

UdpReceiver::~UdpReceiver() {
    close();
    cleanupWinsock();
}

bool UdpReceiver::initWinsock() {
    if (m_wsaInitialized) {
        return true;
    }

    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        m_lastError = "WSAStartup failed with error: " + std::to_string(result);
        return false;
    }

    m_wsaInitialized = true;
    return true;
}

void UdpReceiver::cleanupWinsock() {
    if (m_wsaInitialized) {
        WSACleanup();
        m_wsaInitialized = false;
    }
}

bool UdpReceiver::bind() {
    if (!initWinsock()) {
        return false;
    }

    // Create UDP socket
    m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket == INVALID_SOCKET) {
        m_lastError = "socket() failed with error: " + std::to_string(WSAGetLastError());
        return false;
    }

    // Bind to port
    sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(static_cast<u_short>(m_port));

    if (::bind(m_socket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
        m_lastError = "bind() failed with error: " + std::to_string(WSAGetLastError());
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    return true;
}

void UdpReceiver::close() {
    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
}

std::string UdpReceiver::receive(int timeoutMs) {
    if (m_socket == INVALID_SOCKET) {
        return "";
    }

    // Set receive timeout
    DWORD timeout = static_cast<DWORD>(timeoutMs);
    setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    // Receive data
    char buffer[UDP_BUFFER_SIZE];
    sockaddr_in clientAddr;
    int clientAddrLen = sizeof(clientAddr);

    int bytesReceived = recvfrom(m_socket, buffer, UDP_BUFFER_SIZE - 1, 0,
                                  reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);

    if (bytesReceived == SOCKET_ERROR) {
        int error = WSAGetLastError();
        if (error == WSAETIMEDOUT) {
            // Timeout - normal, return empty string
            return "";
        }
        m_lastError = "recvfrom() failed with error: " + std::to_string(error);
        return "";
    }

    // Null-terminate and return
    buffer[bytesReceived] = '\0';
    return std::string(buffer, bytesReceived);
}

std::string UdpReceiver::getLastError() const {
    return m_lastError;
}
