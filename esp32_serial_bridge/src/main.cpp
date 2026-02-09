#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "serial_port.hpp"
#include "udp_receiver.hpp"
#include "logger.hpp"

// Configuration constants
constexpr int UDP_PORT = 12347;           // TECHNOSOFT_PORT from consts.py
constexpr int SERIAL_BAUD = 115200;
constexpr int UDP_RECV_TIMEOUT_MS = 100;  // For graceful shutdown checks
constexpr int READER_SLEEP_MS = 1;        // Reader loop sleep to avoid busy-wait

// Global shutdown flag
std::atomic<bool> g_running(true);

// Signal handler for Ctrl+C
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
        std::cout << "\nShutdown requested..." << std::endl;
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

// Create directories recursively (Windows)
bool createDirectories(const std::string& path) {
    // Find the last separator to get the directory portion
    size_t lastSep = path.find_last_of("\\/");
    if (lastSep == std::string::npos) {
        return true;  // No directory component
    }

    std::string dirPath = path.substr(0, lastSep);
    if (dirPath.empty()) {
        return true;
    }

    // Try to create each directory component
    std::string current;
    for (size_t i = 0; i < dirPath.size(); ++i) {
        current += dirPath[i];
        if (dirPath[i] == '\\' || dirPath[i] == '/' || i == dirPath.size() - 1) {
            if (!current.empty() && current.back() != ':') {  // Skip drive letter
                CreateDirectoryA(current.c_str(), NULL);
                // Ignore errors - directory may already exist
            }
        }
    }

    // Verify the directory exists
    DWORD attrs = GetFileAttributesA(dirPath.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY));
}

// Generate default log filename with timestamp
std::string generateDefaultLogPath() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::tm tm_now;
    localtime_s(&tm_now, &time_t_now);

    std::ostringstream oss;
    oss << "debug\\esp32_bridge\\"
        << std::put_time(&tm_now, "%Y%m%d_%H%M%S")
        << ".log";

    return oss.str();
}

// Print usage information
void printUsage(const char* programName) {
    std::cout << "ESP32 Serial Bridge - UDP to Serial forwarder\n"
              << "\n"
              << "Usage: " << programName << " <COM_PORT> [OPTIONS]\n"
              << "\n"
              << "Arguments:\n"
              << "  COM_PORT              Serial port (e.g., COM13)\n"
              << "\n"
              << "Options:\n"
              << "  --log-path PATH       Log file path (default: debug\\esp32_bridge\\YYYYMMDD_HHMMSS.log)\n"
              << "  --no-log              Disable all logging (for minimum latency)\n"
              << "  --quiet               Disable console output (file-only logging)\n"
              << "  -h, --help            Show this help message\n"
              << "\n"
              << "Examples:\n"
              << "  " << programName << " COM13\n"
              << "  " << programName << " COM13 --log-path motor_log.txt\n"
              << "  " << programName << " COM13 --no-log\n"
              << "  " << programName << " COM13 --quiet\n"
              << std::endl;
}

// Parse command line arguments
struct Config {
    std::string comPort;
    std::string logPath;
    bool enableConsole = true;
    bool enableFile = true;
    bool valid = false;
};

Config parseArgs(int argc, char* argv[]) {
    Config config;

    if (argc < 2) {
        return config;
    }

    // First argument is COM port
    std::string firstArg = argv[1];
    if (firstArg == "-h" || firstArg == "--help") {
        printUsage(argv[0]);
        exit(0);
    }

    config.comPort = firstArg;
    config.logPath = generateDefaultLogPath();

    // Parse optional arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--log-path" && i + 1 < argc) {
            config.logPath = argv[++i];
        }
        else if (arg == "--no-log") {
            config.enableConsole = false;
            config.enableFile = false;
        }
        else if (arg == "--quiet") {
            config.enableConsole = false;
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return config;
        }
    }

    config.valid = true;
    return config;
}

// Writer thread: receives UDP commands and forwards to serial
void writerThread(SerialPort& serial, UdpReceiver& udp, Logger& logger) {
    logger.logInfo("Writer thread started");
    std::string lastMessage;  // Track last sent message to avoid duplicates

    while (g_running) {
        // Receive UDP packet with timeout
        std::string data = udp.receive(UDP_RECV_TIMEOUT_MS);

        if (!data.empty()) {
            // Log received UDP command
            logger.logUdpReceived(data);

            // Skip if same as last message (optimization to reduce serial traffic)
            if (data == lastMessage) {
                continue;
            }

            // Forward to serial immediately
            if (serial.write(data)) {
                logger.logSerialSent(data);
                lastMessage = data;
            } else {
                logger.logError("Serial write failed: " + serial.getLastError());
            }
        }
    }

    logger.logInfo("Writer thread stopped");
}

// Reader thread: reads serial responses and logs them
void readerThread(SerialPort& serial, Logger& logger) {
    logger.logInfo("Reader thread started");

    while (g_running) {
        // Read available serial data
        std::string data = serial.readAvailable();

        if (!data.empty()) {
            // Remove trailing newlines for cleaner logging
            while (!data.empty() && (data.back() == '\n' || data.back() == '\r')) {
                data.pop_back();
            }

            if (!data.empty()) {
                logger.logSerialReceived(data);
            }
        }

        // Small sleep to avoid busy-wait
        std::this_thread::sleep_for(std::chrono::milliseconds(READER_SLEEP_MS));
    }

    logger.logInfo("Reader thread stopped");
}

int main(int argc, char* argv[]) {
    // Parse arguments
    Config config = parseArgs(argc, argv);

    if (!config.valid) {
        printUsage(argv[0]);
        return 1;
    }

    // Set up signal handler for Ctrl+C
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
        std::cerr << "Warning: Could not set up signal handler" << std::endl;
    }

    // Create log directory if it doesn't exist
    if (config.enableFile && !createDirectories(config.logPath)) {
        std::cerr << "Warning: Could not create log directory for: " << config.logPath << std::endl;
    }

    // Initialize logger
    Logger logger(config.logPath, config.enableConsole, config.enableFile);

    std::cout << "ESP32 Serial Bridge" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "COM Port: " << config.comPort << std::endl;
    std::cout << "UDP Port: " << UDP_PORT << std::endl;
    if (config.enableFile) {
        std::cout << "Log File: " << config.logPath << std::endl;
    }
    std::cout << "Press Ctrl+C to exit" << std::endl;
    std::cout << std::endl;

    logger.logInfo("Starting ESP32 Serial Bridge");
    logger.logInfo("COM Port: " + config.comPort);
    logger.logInfo("UDP Port: " + std::to_string(UDP_PORT));

    // Initialize serial port
    SerialPort serial(config.comPort, SERIAL_BAUD);
    if (!serial.open()) {
        logger.logError("Failed to open serial port: " + serial.getLastError());
        std::cerr << "Error: " << serial.getLastError() << std::endl;
        return 1;
    }
    logger.logInfo("Serial port opened successfully");

    // Initialize UDP receiver
    UdpReceiver udp(UDP_PORT);
    if (!udp.bind()) {
        logger.logError("Failed to bind UDP socket: " + udp.getLastError());
        std::cerr << "Error: " << udp.getLastError() << std::endl;
        return 1;
    }
    logger.logInfo("UDP socket bound to port " + std::to_string(UDP_PORT));

    // Start worker threads
    std::thread writer(writerThread, std::ref(serial), std::ref(udp), std::ref(logger));
    std::thread reader(readerThread, std::ref(serial), std::ref(logger));

    logger.logInfo("Bridge is running");

    // Wait for threads to finish
    writer.join();
    reader.join();

    // Cleanup
    logger.logInfo("Shutting down...");
    udp.close();
    serial.close();
    logger.logInfo("ESP32 Serial Bridge stopped");

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
