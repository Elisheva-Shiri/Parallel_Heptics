#pragma once

#include <string>
#include <windows.h>

class SerialPort {
public:
    SerialPort(const std::string& portName, int baudRate = 115200);
    ~SerialPort();

    // Disable copy
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    bool open();
    void close();
    bool isOpen() const;

    // Non-blocking write - sends data to hardware buffer
    bool write(const std::string& data);
    bool write(const char* data, size_t length);

    // Non-blocking read - returns available data or empty string
    std::string readAvailable();

    // Get last error for diagnostics
    std::string getLastError() const;

private:
    std::string m_portName;
    int m_baudRate;
    HANDLE m_handle;
    std::string m_lastError;

    bool configurePort();
};
