#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <atomic>

enum class LogDirection {
    UDP_IN,
    SERIAL_OUT,
    SERIAL_IN,
    INFO,
    ERR  // Not ERROR - conflicts with Windows macro
};

class Logger {
public:
    Logger(const std::string& logPath, bool enableConsole, bool enableFile);
    ~Logger();

    // Disable copy
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Thread-safe logging methods
    void log(LogDirection direction, const std::string& message);
    void logUdpReceived(const std::string& message);
    void logSerialSent(const std::string& message);
    void logSerialReceived(const std::string& message);
    void logInfo(const std::string& message);
    void logError(const std::string& message);

    // Check if logging is enabled
    bool isEnabled() const;

private:
    std::ofstream m_fileStream;
    std::mutex m_mutex;
    bool m_enableConsole;
    bool m_enableFile;

    std::string formatTimestamp();
    std::string directionToString(LogDirection dir);
    void writeLog(const std::string& formattedMessage);
};
