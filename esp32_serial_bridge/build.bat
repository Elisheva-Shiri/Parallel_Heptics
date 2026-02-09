@echo off
setlocal

:: ESP32 Serial Bridge - MinGW Build Script
:: Requires: MinGW with g++ in PATH

echo ========================================
echo ESP32 Serial Bridge - MinGW Build
echo ========================================
echo.

:: Check if g++ is available
where g++ >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: g++ not found in PATH
    echo Please install MinGW and add it to your PATH
    exit /b 1
)

:: Configuration
set CXX=g++
set CXXFLAGS=-std=c++17 -O2 -Wall -Wextra -DWIN32_LEAN_AND_MEAN
set LDFLAGS=-lws2_32 -static -static-libgcc -static-libstdc++

:: Source files
set SOURCES=src\main.cpp src\serial_port.cpp src\udp_receiver.cpp src\logger.cpp

:: Output
set OUTPUT=esp32_bridge.exe

echo Compiler: %CXX%
echo Sources: %SOURCES%
echo Output: %OUTPUT%
echo.

echo Compiling...
%CXX% %CXXFLAGS% %SOURCES% -o %OUTPUT% %LDFLAGS%

if %ERRORLEVEL% == 0 (
    echo.
    echo ========================================
    echo Build successful: %OUTPUT%
    echo ========================================
    echo.
    echo Usage:
    echo   %OUTPUT% COM13
    echo   %OUTPUT% COM13 --log-path motor_log.txt
    echo   %OUTPUT% COM13 --no-log
) else (
    echo.
    echo ========================================
    echo Build FAILED!
    echo ========================================
    exit /b 1
)

endlocal
