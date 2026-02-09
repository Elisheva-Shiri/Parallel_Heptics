#include "serial_port.hpp"
#include <iostream>

SerialPort::SerialPort(const std::string& portName, int baudRate)
    : m_portName(portName)
    , m_baudRate(baudRate)
    , m_handle(INVALID_HANDLE_VALUE)
{
}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open() {
    // Windows COM port requires \\.\ prefix for COM10 and above
    std::string fullPortName = "\\\\.\\" + m_portName;

    m_handle = CreateFileA(
        fullPortName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,              // No sharing
        nullptr,        // Default security
        OPEN_EXISTING,
        0,              // No overlapped I/O
        nullptr         // No template
    );

    if (m_handle == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        m_lastError = "Failed to open " + m_portName + " (error " + std::to_string(error) + ")";
        return false;
    }

    if (!configurePort()) {
        close();
        return false;
    }

    return true;
}

bool SerialPort::configurePort() {
    // Get current state
    DCB dcb;
    ZeroMemory(&dcb, sizeof(dcb));
    dcb.DCBlength = sizeof(DCB);

    if (!GetCommState(m_handle, &dcb)) {
        m_lastError = "Failed to get comm state";
        return false;
    }

    // Configure serial parameters: 115200 8N1
    dcb.BaudRate = m_baudRate;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;

    // CRITICAL: Disable DTR/RTS to prevent ESP32 reset
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDsrSensitivity = FALSE;

    // Disable XON/XOFF flow control
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;

    // Other settings
    dcb.fBinary = TRUE;
    dcb.fNull = FALSE;
    dcb.fAbortOnError = FALSE;
    dcb.fErrorChar = FALSE;

    if (!SetCommState(m_handle, &dcb)) {
        m_lastError = "Failed to set comm state";
        return false;
    }

    // Set timeouts for non-blocking reads
    COMMTIMEOUTS timeouts;
    ZeroMemory(&timeouts, sizeof(timeouts));
    timeouts.ReadIntervalTimeout = MAXDWORD;        // Return immediately
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.ReadTotalTimeoutConstant = 0;          // No total timeout
    timeouts.WriteTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = 1000;      // 1 second write timeout

    if (!SetCommTimeouts(m_handle, &timeouts)) {
        m_lastError = "Failed to set comm timeouts";
        return false;
    }

    // Clear any existing data in buffers
    PurgeComm(m_handle, PURGE_RXCLEAR | PURGE_TXCLEAR);

    return true;
}

void SerialPort::close() {
    if (m_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_handle);
        m_handle = INVALID_HANDLE_VALUE;
    }
}

bool SerialPort::isOpen() const {
    return m_handle != INVALID_HANDLE_VALUE;
}

bool SerialPort::write(const std::string& data) {
    return write(data.c_str(), data.length());
}

bool SerialPort::write(const char* data, size_t length) {
    if (!isOpen()) {
        m_lastError = "Port not open";
        return false;
    }

    DWORD bytesWritten = 0;
    if (!WriteFile(m_handle, data, static_cast<DWORD>(length), &bytesWritten, nullptr)) {
        DWORD error = GetLastError();
        m_lastError = "Write failed (error " + std::to_string(error) + ")";
        return false;
    }

    if (bytesWritten != length) {
        m_lastError = "Incomplete write: " + std::to_string(bytesWritten) + "/" + std::to_string(length);
        return false;
    }

    return true;
}

std::string SerialPort::readAvailable() {
    if (!isOpen()) {
        return "";
    }

    // Check how many bytes are available
    COMSTAT comStat;
    DWORD errors;
    if (!ClearCommError(m_handle, &errors, &comStat)) {
        return "";
    }

    if (comStat.cbInQue == 0) {
        return "";
    }

    // Read available data
    std::string buffer(comStat.cbInQue, '\0');
    DWORD bytesRead = 0;

    if (!ReadFile(m_handle, &buffer[0], comStat.cbInQue, &bytesRead, nullptr)) {
        return "";
    }

    buffer.resize(bytesRead);
    return buffer;
}

std::string SerialPort::getLastError() const {
    return m_lastError;
}
