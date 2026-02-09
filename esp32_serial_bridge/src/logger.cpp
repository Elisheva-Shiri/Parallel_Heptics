#include "logger.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <windows.h>

Logger::Logger(const std::string& logPath, bool enableConsole, bool enableFile)
    : m_enableConsole(enableConsole)
    , m_enableFile(enableFile)
{
    if (m_enableFile && !logPath.empty()) {
        m_fileStream.open(logPath, std::ios::out | std::ios::app);
        if (!m_fileStream.is_open()) {
            std::cerr << "Warning: Failed to open log file: " << logPath << std::endl;
            m_enableFile = false;
        }
    }
}

Logger::~Logger() {
    if (m_fileStream.is_open()) {
        m_fileStream.close();
    }
}

std::string Logger::formatTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_now;
    localtime_s(&tm_now, &time_t_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

std::string Logger::directionToString(LogDirection dir) {
    switch (dir) {
        case LogDirection::UDP_IN:     return "UDP IN    ";
        case LogDirection::SERIAL_OUT: return "SERIAL OUT";
        case LogDirection::SERIAL_IN:  return "SERIAL IN ";
        case LogDirection::INFO:       return "INFO      ";
        case LogDirection::ERR:        return "ERROR     ";
        default:                       return "UNKNOWN   ";
    }
}

void Logger::writeLog(const std::string& formattedMessage) {
    if (m_enableConsole) {
        std::cout << formattedMessage << std::endl;
    }

    if (m_enableFile && m_fileStream.is_open()) {
        m_fileStream << formattedMessage << std::endl;
        m_fileStream.flush();
    }
}

void Logger::log(LogDirection direction, const std::string& message) {
    if (!m_enableConsole && !m_enableFile) {
        return;
    }

    std::string timestamp = formatTimestamp();
    std::string dirStr = directionToString(direction);

    std::ostringstream oss;
    oss << "[" << timestamp << "] [" << dirStr << "] " << message;

    std::lock_guard<std::mutex> lock(m_mutex);
    writeLog(oss.str());
}

void Logger::logUdpReceived(const std::string& message) {
    log(LogDirection::UDP_IN, message);
}

void Logger::logSerialSent(const std::string& message) {
    log(LogDirection::SERIAL_OUT, message);
}

void Logger::logSerialReceived(const std::string& message) {
    log(LogDirection::SERIAL_IN, message);
}

void Logger::logInfo(const std::string& message) {
    log(LogDirection::INFO, message);
}

void Logger::logError(const std::string& message) {
    log(LogDirection::ERR, message);
}

bool Logger::isEnabled() const {
    return m_enableConsole || m_enableFile;
}
