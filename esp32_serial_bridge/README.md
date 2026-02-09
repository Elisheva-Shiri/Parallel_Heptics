# ESP32 Serial Bridge

A high-performance C++ UDP-to-Serial bridge for ESP32 motor control. Receives motor commands via UDP and forwards them directly to the ESP32 via serial, minimizing latency compared to Python serial communication.

## Prerequisites

You need **one** of the following C++ compilers installed:

### Option 1: MinGW (Recommended for simplicity)

1. Download MinGW-w64 from: https://www.mingw-w64.org/downloads/
   - Or use MSYS2: https://www.msys2.org/

2. If using MSYS2, open MSYS2 terminal and run:
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   ```

3. Add MinGW bin folder to your PATH:
   - Default MSYS2 path: `C:\msys64\mingw64\bin`
   - Verify installation: `g++ --version`

### Option 2: Visual Studio

1. Download Visual Studio from: https://visualstudio.microsoft.com/
   - Community edition is free

2. During installation, select **"Desktop development with C++"** workload

3. After installation, the build script will automatically find Visual Studio

## Building

Open a **Command Prompt** (not PowerShell) in this folder and run:

```batch
# Using MinGW
build.bat

# Using Visual Studio
build_msvc.bat
```

This creates `esp32_bridge.exe` in the current folder.

## Usage

```
esp32_bridge.exe <COM_PORT> [OPTIONS]

Arguments:
  COM_PORT              Serial port (e.g., COM13)

Options:
  --log-path PATH       Log file path (default: esp32_bridge_YYYYMMDD_HHMMSS.log)
  --no-log              Disable all logging (for minimum latency)
  --quiet               Disable console output (file-only logging)
  -h, --help            Show help message
```

### Examples

```batch
# Basic usage with default logging
esp32_bridge.exe COM13

# Custom log file
esp32_bridge.exe COM13 --log-path motor_log.txt

# No logging for minimum latency
esp32_bridge.exe COM13 --no-log

# Log to file only, no console output
esp32_bridge.exe COM13 --quiet
```

### Stopping

Press `Ctrl+C` to gracefully shut down the bridge.

## Integration with Python Backend

The bridge replaces direct Python-to-ESP32 serial communication with a UDP intermediary:

```
Python Backend  ---(UDP:12347)--->  ESP32 Bridge  ---(Serial)--->  ESP32
```

### Setup

1. Start the bridge first:
   ```batch
   esp32_bridge.exe COM13
   ```

2. In `backend.py`, change line 46:
   ```python
   MOTOR_TYPE = MotorType.TECHNOSOFT  # Was: MotorType.ARDUINO
   ```

3. Run the Python backend normally - it will send UDP packets to port 12347, which the bridge forwards to serial.

## Protocol

### Message Format

Motor commands use ASCII format: `ZM<index>P<position>...F`

Examples:
- `ZM0P500F` - Move motor 0 to position 500
- `ZM0P100M1P-50M2P200F` - Move motors 0, 1, 2 simultaneously

### Communication

- **UDP Port:** 12347 (matches `TECHNOSOFT_PORT` in `consts.py`)
- **Serial:** 115200 baud, 8N1, no flow control
- **DTR/RTS:** Disabled to prevent ESP32 reset on connection

## Log Format

```
[2026-01-25 12:34:56.789] [UDP IN    ] ZM0P500M1P-200F
[2026-01-25 12:34:56.790] [SERIAL OUT] ZM0P500M1P-200F
[2026-01-25 12:34:56.823] [SERIAL IN ] OK:M0P500,M1P-200
```

## Architecture

```
┌─────────────────┐
│   Main Thread   │  Initialization, signal handling, cleanup
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Writer │ │Reader │
│Thread │ │Thread │
└───┬───┘ └───┬───┘
    │         │
    │  UDP    │  Serial
    │  recv   │  read
    │    │    │    │
    │    ▼    │    ▼
    │ Serial  │  Logger
    │  write  │
    └────┬────┘
         │
    ┌────▼────┐
    │  ESP32  │
    └─────────┘
```

## Troubleshooting

### "g++ not found"
- Install MinGW and add to PATH (see Prerequisites)
- Or use `build_msvc.bat` with Visual Studio instead

### "Failed to open COM port"
- Check the port name (use Device Manager to find correct COM port)
- Ensure no other program is using the port
- Try running as Administrator

### "Failed to bind UDP socket"
- Port 12347 may be in use - check for other instances
- Firewall may be blocking - allow the program through Windows Firewall

### ESP32 resets on connection
- This shouldn't happen (DTR/RTS are disabled)
- If it does, check your USB cable and ESP32 board

## Files

```
esp32_serial_bridge/
├── src/
│   ├── main.cpp           # Entry point, threads, CLI parsing
│   ├── serial_port.hpp    # Serial port interface
│   ├── serial_port.cpp    # Windows COM port implementation
│   ├── udp_receiver.hpp   # UDP receiver interface
│   ├── udp_receiver.cpp   # Winsock implementation
│   ├── logger.hpp         # Logger interface
│   └── logger.cpp         # Thread-safe logging
├── build.bat              # MinGW build script
├── build_msvc.bat         # MSVC build script
└── README.md              # This file
```
