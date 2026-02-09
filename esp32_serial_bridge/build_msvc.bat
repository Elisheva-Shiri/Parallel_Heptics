@echo off
setlocal

:: ESP32 Serial Bridge - MSVC Build Script
:: Requires: Visual Studio with C++ workload installed

echo ========================================
echo ESP32 Serial Bridge - MSVC Build
echo ========================================
echo.

:: Try to find Visual Studio
set "VS_FOUND="

:: Try VS 2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2022 Community
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)
if not defined VS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2022 Professional
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)
if not defined VS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2022 Enterprise
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)

:: Try VS 2019
if not defined VS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2019 Community
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)
if not defined VS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2019 Professional
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)
if not defined VS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2019 Enterprise
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)

:: Try Build Tools
if not defined VS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2022 Build Tools
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)
if not defined VS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio 2019 Build Tools
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
)

if not defined VS_FOUND (
    echo ERROR: Visual Studio not found!
    echo Please install Visual Studio with C++ workload
    echo or use build.bat with MinGW instead.
    exit /b 1
)

echo.

:: Configuration
set CXXFLAGS=/std:c++17 /O2 /W4 /EHsc /DWIN32_LEAN_AND_MEAN /DNOMINMAX
set LDFLAGS=ws2_32.lib

:: Source files
set SOURCES=src\main.cpp src\serial_port.cpp src\udp_receiver.cpp src\logger.cpp

:: Output
set OUTPUT=esp32_bridge.exe

echo Compiler: cl.exe (MSVC)
echo Sources: %SOURCES%
echo Output: %OUTPUT%
echo.

echo Compiling...
cl %CXXFLAGS% %SOURCES% /Fe:%OUTPUT% /link %LDFLAGS%

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

    :: Clean up object files
    del *.obj >nul 2>&1
) else (
    echo.
    echo ========================================
    echo Build FAILED!
    echo ========================================
    exit /b 1
)

endlocal
