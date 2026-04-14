/**
 * ESP32 Dual-Core Servo Controller via PCA9685
 *
 * Hardware:
 *   - ESP32
 *   - PCA9685 16-channel 12-bit PWM/Servo driver (I2C)
 *   - 3x SG92R servo motors on PCA9685 channels 0-2
 *
 * Core 0: Receives serial commands, parses them, updates command queue
 * Core 1: Executes synchronized servo movements with easing
 *
 * Message format: "ZM<idx>P<pos>M<idx>P<pos>...F"
 * Example: "ZM0P100M1P-50M2P75F"
 *
 * Based on esp32_motor_controller with PCA9685 HAL and motion blending from v13.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ============================================================================
// Configuration
// ============================================================================

#define NUM_MOTORS 3           // 3 servos on PCA9685 channels 0-2
#define SERIAL_BAUD 115200
#define MAX_MESSAGE_LENGTH 128
#define COMMAND_QUEUE_SIZE 10

// PCA9685 I2C configuration
#define I2C_SDA 21
#define I2C_SCL 22
#define PCA9685_ADDRESS 0x40
#define PCA9685_FREQ 50        // 50 Hz for standard servos

// Servo channel mapping on PCA9685
const int SERVO_CHANNELS[NUM_MOTORS] = {0, 1, 2};

// Servo pulse range in PCA9685 ticks (12-bit, 4096 ticks per cycle)
// At 50 Hz (20ms period): 1 tick = 20000/4096 ≈ 4.88 µs
// SG92R typical range: 500-2400 µs → ~130-500 ticks
// Calibrate these per servo using the CAL command
int servoMinTick[NUM_MOTORS] = {130, 130, 130};
int servoMaxTick[NUM_MOTORS] = {500, 500, 500};

// Position mapping
const int POSITION_RANGE = 1000;  // Backend sends positions in range [-1000, 1000]

// Deadband: minimum tick change to actually write to PCA9685 (reduces jitter)
// Set to 2+ for SG92R jitter suppression; 1 = redundant-write filter only
#define SERVO_DEADBAND_TICKS 2

// Movement configuration
#define MOVEMENT_UPDATE_INTERVAL_MS 5   // Fast updates for smooth blending at 150 Hz
#define MOVEMENT_TIMEOUT_MS 2000        // Safety timeout for movement loop
#define BLEND_ENABLED true              // Enable motion blending for continuous command streams

// ============================================================================
// Data Structures
// ============================================================================

struct MotorCommand {
    int motorIndex;
    int targetPosition;
};

struct CommandSet {
    MotorCommand commands[NUM_MOTORS];
    int commandCount;
    bool isValid;
};

// ============================================================================
// Forward Declarations
// ============================================================================

bool dequeueCommandSet(CommandSet* cmdSet);
bool hasQueuedCommand();
void processDebugCommand(const char* cmd);

// ============================================================================
// Global Variables
// ============================================================================

// PCA9685 driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDRESS);

// Motor state
volatile int currentPositions[NUM_MOTORS] = {0};
volatile int targetPositions[NUM_MOTORS] = {0};
int lastWrittenTicks[NUM_MOTORS] = {0};  // Track last tick written to avoid redundant writes
volatile bool motorsMoving = false;

// Command queue (thread-safe communication between cores)
CommandSet commandQueue[COMMAND_QUEUE_SIZE];
int queueHead = 0;
int queueTail = 0;
portMUX_TYPE queueMux = portMUX_INITIALIZER_UNLOCKED;

// Serial buffer
char serialBuffer[MAX_MESSAGE_LENGTH];
int bufferIndex = 0;
bool messageStarted = false;

// Task handles
TaskHandle_t serialTaskHandle;
TaskHandle_t motorTaskHandle;

// ============================================================================
// Serial Output Helper (sends \r\n for proper terminal display)
// ============================================================================

void serialPrintln(const char* msg) {
    Serial.print(msg);
    Serial.print("\r\n");
}

void serialPrintf(const char* format, ...) {
    char buf[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    Serial.print(buf);
}

// ============================================================================
// Motor Abstraction Layer (PCA9685)
// ============================================================================

/**
 * Convert a backend position to a PCA9685 tick value for a specific servo.
 */
int positionToTick(int motorIndex, int position) {
    int pos = constrain(position, -POSITION_RANGE, POSITION_RANGE);

    // Map from [-POSITION_RANGE, +POSITION_RANGE] to [servoMinTick, servoMaxTick]
    int tick = map((long)pos, (long)-POSITION_RANGE, (long)POSITION_RANGE,
                   (long)servoMinTick[motorIndex], (long)servoMaxTick[motorIndex]);
    return constrain(tick, servoMinTick[motorIndex], servoMaxTick[motorIndex]);
}

/**
 * Initialize the PCA9685 servo driver.
 */
void initializeMotorHardware() {
    Wire.begin(I2C_SDA, I2C_SCL);
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(PCA9685_FREQ);
    delay(10);

    // Center all servos at position 0
    for (int i = 0; i < NUM_MOTORS; i++) {
        currentPositions[i] = 0;
        targetPositions[i] = 0;
        int tick = positionToTick(i, 0);
        pwm.setPWM(SERVO_CHANNELS[i], 0, tick);
        lastWrittenTicks[i] = tick;
    }
}

/**
 * Set servo to a position value via PCA9685.
 * Applies deadband to avoid jitter from redundant writes.
 *
 * @param motorIndex Index of the servo (0 to NUM_MOTORS-1)
 * @param position Position value (-POSITION_RANGE to +POSITION_RANGE)
 */
void setMotorPositionRaw(int motorIndex, int position) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return;

    int tick = positionToTick(motorIndex, position);

    // Deadband: only write if tick actually changed
    if (abs(tick - lastWrittenTicks[motorIndex]) >= SERVO_DEADBAND_TICKS) {
        pwm.setPWM(SERVO_CHANNELS[motorIndex], 0, tick);
        lastWrittenTicks[motorIndex] = tick;
    }
}

/**
 * Cleanup: turn off all servo PWM outputs (servos go limp).
 */
void cleanupMotorHardware() {
    for (int i = 0; i < NUM_MOTORS; i++) {
        pwm.setPWM(SERVO_CHANNELS[i], 0, 4096);  // Full-off
    }
}

// ============================================================================
// Logical Motor Control Layer
// ============================================================================

/**
 * Move a motor to a target position.
 */
void moveMotorToPosition(int motorIndex, int position) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return;

    currentPositions[motorIndex] = position;
    setMotorPositionRaw(motorIndex, position);
}

/**
 * Easing function for smooth movement (ease-in-out quadratic).
 */
float easeInOutQuad(float t) {
    return t < 0.5 ? 2 * t * t : 1 - pow(-2 * t + 2, 2) / 2;
}

/**
 * Execute a synchronized movement for multiple motors.
 * All motors start and finish at the same time regardless of travel distance.
 * Supports motion blending: if a new command arrives mid-movement, the trajectory
 * is recalculated from current interpolated positions.
 *
 * @param commands Array of motor commands
 * @param commandCount Number of commands in the array
 */
void executeSynchronizedMovement(MotorCommand* commands, int commandCount) {
    if (commandCount == 0) return;

    // Active command tracking (updated on blend to avoid stale final-position snap)
    MotorCommand activeCommands[NUM_MOTORS];
    int activeCommandCount = 0;

    // Copy initial commands
    for (int i = 0; i < commandCount && i < NUM_MOTORS; i++) {
        activeCommands[i] = commands[i];
    }
    activeCommandCount = min(commandCount, NUM_MOTORS);

    // Store starting positions and calculate deltas
    int startPositions[NUM_MOTORS];
    int deltaPositions[NUM_MOTORS];
    int involvedIndices[NUM_MOTORS];
    int involvedCount = 0;

    int maxDistance = 0;
    for (int i = 0; i < activeCommandCount; i++) {
        int idx = activeCommands[i].motorIndex;
        if (idx >= 0 && idx < NUM_MOTORS) {
            startPositions[idx] = currentPositions[idx];
            deltaPositions[idx] = activeCommands[i].targetPosition - startPositions[idx];
            involvedIndices[involvedCount++] = idx;

            int dist = abs(deltaPositions[idx]);
            if (dist > maxDistance) maxDistance = dist;
        }
    }

    if (involvedCount == 0) return;

    // Calculate movement duration: scale with distance, minimum 10ms
    // At 150 Hz commands arrive every ~6.7ms, so keep durations short
    int durationMs = max(10, maxDistance / 10);

    motorsMoving = true;
    unsigned long startTime = millis();

    // Main control loop with blending support
    while (motorsMoving) {
        // Safety timeout
        if (millis() - startTime > MOVEMENT_TIMEOUT_MS) {
            serialPrintln("E:TIMEOUT");
            break;
        }

        unsigned long elapsed = millis() - startTime;
        float progress = (float)elapsed / (float)durationMs;
        if (progress > 1.0) progress = 1.0;

        // Check for motion blending - if new command available, blend to it
        if (BLEND_ENABLED && hasQueuedCommand()) {
            CommandSet newCmd;
            if (dequeueCommandSet(&newCmd)) {
                if (newCmd.isValid && newCmd.commandCount > 0) {
                    // Update active commands to the blended set
                    activeCommandCount = 0;
                    involvedCount = 0;
                    maxDistance = 0;

                    for (int i = 0; i < newCmd.commandCount && i < NUM_MOTORS; i++) {
                        int idx = newCmd.commands[i].motorIndex;
                        if (idx >= 0 && idx < NUM_MOTORS) {
                            activeCommands[activeCommandCount++] = newCmd.commands[i];
                            startPositions[idx] = currentPositions[idx];
                            deltaPositions[idx] = newCmd.commands[i].targetPosition - startPositions[idx];
                            involvedIndices[involvedCount++] = idx;

                            int dist = abs(deltaPositions[idx]);
                            if (dist > maxDistance) maxDistance = dist;
                        }
                    }

                    // Reset timing for blended trajectory
                    durationMs = max(10, maxDistance / 10);
                    startTime = millis();
                    continue;
                }
            }
        }

        // Apply easing and update positions
        float easedProgress = easeInOutQuad(progress);

        for (int i = 0; i < involvedCount; i++) {
            int idx = involvedIndices[i];
            int newPosition = startPositions[idx] + (int)(deltaPositions[idx] * easedProgress);
            moveMotorToPosition(idx, newPosition);
        }

        // Movement complete
        if (progress >= 1.0) {
            // Ensure final positions are exact (using active commands, not original)
            for (int i = 0; i < activeCommandCount; i++) {
                int idx = activeCommands[i].motorIndex;
                if (idx >= 0 && idx < NUM_MOTORS) {
                    moveMotorToPosition(idx, activeCommands[i].targetPosition);
                }
            }
            motorsMoving = false;
            break;
        }

        delay(MOVEMENT_UPDATE_INTERVAL_MS);
    }

    motorsMoving = false;
}

// ============================================================================
// Command Queue Functions
// ============================================================================

bool enqueueCommandSet(CommandSet* cmdSet) {
    portENTER_CRITICAL(&queueMux);

    int nextTail = (queueTail + 1) % COMMAND_QUEUE_SIZE;
    if (nextTail == queueHead) {
        portEXIT_CRITICAL(&queueMux);
        return false;
    }

    commandQueue[queueTail] = *cmdSet;
    queueTail = nextTail;

    portEXIT_CRITICAL(&queueMux);
    return true;
}

bool dequeueCommandSet(CommandSet* cmdSet) {
    portENTER_CRITICAL(&queueMux);

    if (queueHead == queueTail) {
        portEXIT_CRITICAL(&queueMux);
        return false;
    }

    *cmdSet = commandQueue[queueHead];
    queueHead = (queueHead + 1) % COMMAND_QUEUE_SIZE;

    portEXIT_CRITICAL(&queueMux);
    return true;
}

bool hasQueuedCommand() {
    portENTER_CRITICAL(&queueMux);
    bool hasCommand = (queueHead != queueTail);
    portEXIT_CRITICAL(&queueMux);
    return hasCommand;
}

// ============================================================================
// Message Parsing
// ============================================================================

bool parseMessage(const char* message, CommandSet* cmdSet) {
    cmdSet->commandCount = 0;
    cmdSet->isValid = false;

    if (message[0] != 'Z') {
        return false;
    }

    int i = 1;
    int len = strlen(message);

    while (i < len && message[i] != 'F') {
        if (message[i] != 'M') {
            i++;
            continue;
        }
        i++;

        // Parse motor index
        int motorIndex = 0;
        while (i < len && message[i] >= '0' && message[i] <= '9') {
            motorIndex = motorIndex * 10 + (message[i] - '0');
            i++;
        }

        if (i >= len || message[i] != 'P') {
            return false;
        }
        i++;

        // Parse position (may be negative)
        int sign = 1;
        if (message[i] == '-') {
            sign = -1;
            i++;
        }

        int position = 0;
        while (i < len && message[i] >= '0' && message[i] <= '9') {
            position = position * 10 + (message[i] - '0');
            i++;
        }
        position *= sign;

        if (cmdSet->commandCount < NUM_MOTORS) {
            cmdSet->commands[cmdSet->commandCount].motorIndex = motorIndex;
            cmdSet->commands[cmdSet->commandCount].targetPosition = position;
            cmdSet->commandCount++;
        }
    }

    if (i < len && message[i] == 'F') {
        cmdSet->isValid = true;
        return true;
    }

    return false;
}

// ============================================================================
// Debug Commands
// ============================================================================

void processDebugCommand(const char* cmd) {
    if (strncmp(cmd, "STATUS", 6) == 0) {
        serialPrintln("=== Servo Status ===");
        for (int i = 0; i < NUM_MOTORS; i++) {
            serialPrintf("Servo %d: Pos=%d, Tick=%d (last written), Range=[%d-%d]\r\n",
                i, currentPositions[i], lastWrittenTicks[i],
                servoMinTick[i], servoMaxTick[i]);
        }
    }
    else if (strncmp(cmd, "RESET", 5) == 0) {
        for (int i = 0; i < NUM_MOTORS; i++) {
            moveMotorToPosition(i, 0);
        }
        serialPrintln("All servos centered (position 0)");
    }
    else if (strncmp(cmd, "CAL", 3) == 0) {
        // CAL <channel> <tick> - Set a specific PCA9685 tick value for calibration
        // CAL <channel> MIN <tick> - Set minimum tick for channel
        // CAL <channel> MAX <tick> - Set maximum tick for channel
        // CAL SHOW - Show current calibration values
        int channel, tick;
        char subcmd[8];

        if (strncmp(cmd + 3, " SHOW", 5) == 0 || strlen(cmd) == 3) {
            serialPrintln("=== Calibration Values ===");
            for (int i = 0; i < NUM_MOTORS; i++) {
                serialPrintf("Servo %d: MIN=%d, MAX=%d (range=%d ticks)\r\n",
                    i, servoMinTick[i], servoMaxTick[i],
                    servoMaxTick[i] - servoMinTick[i]);
            }
        }
        else if (sscanf(cmd, "CAL %d MIN %d", &channel, &tick) == 2) {
            if (channel >= 0 && channel < NUM_MOTORS && tick >= 0 && tick < 4096) {
                servoMinTick[channel] = tick;
                serialPrintf("Servo %d MIN set to %d\r\n", channel, tick);
                if (servoMinTick[channel] >= servoMaxTick[channel]) {
                    serialPrintf("WARNING: MIN(%d) >= MAX(%d) for servo %d\r\n",
                        servoMinTick[channel], servoMaxTick[channel], channel);
                }
            } else {
                serialPrintln("E:Invalid channel or tick value");
            }
        }
        else if (sscanf(cmd, "CAL %d MAX %d", &channel, &tick) == 2) {
            if (channel >= 0 && channel < NUM_MOTORS && tick >= 0 && tick < 4096) {
                servoMaxTick[channel] = tick;
                serialPrintf("Servo %d MAX set to %d\r\n", channel, tick);
                if (servoMinTick[channel] >= servoMaxTick[channel]) {
                    serialPrintf("WARNING: MIN(%d) >= MAX(%d) for servo %d\r\n",
                        servoMinTick[channel], servoMaxTick[channel], channel);
                }
            } else {
                serialPrintln("E:Invalid channel or tick value");
            }
        }
        else if (sscanf(cmd, "CAL %d %d", &channel, &tick) == 2) {
            if (channel >= 0 && channel < NUM_MOTORS && tick >= 0 && tick < 4096) {
                pwm.setPWM(SERVO_CHANNELS[channel], 0, tick);
                lastWrittenTicks[channel] = tick;
                serialPrintf("Servo %d set to tick %d\r\n", channel, tick);
            } else {
                serialPrintln("E:Invalid channel or tick value");
            }
        }
        else {
            serialPrintln("Usage: CAL [SHOW]");
            serialPrintln("       CAL <ch> <tick>       - Move servo to tick");
            serialPrintln("       CAL <ch> MIN <tick>   - Set min tick");
            serialPrintln("       CAL <ch> MAX <tick>   - Set max tick");
        }
    }
    else if (strncmp(cmd, "HELP", 4) == 0) {
        serialPrintln("=== Commands ===");
        serialPrintln("STATUS              - Show servo status");
        serialPrintln("RESET               - Center all servos");
        serialPrintln("CAL [SHOW]          - Show calibration");
        serialPrintln("CAL <ch> <tick>     - Move servo to tick");
        serialPrintln("CAL <ch> MIN <tick> - Set min tick");
        serialPrintln("CAL <ch> MAX <tick> - Set max tick");
        serialPrintln("HELP                - Show this help");
        serialPrintln("ZM<n>P<pos>F        - Move servo n to position");
    }
    else {
        serialPrintf("Unknown command: %s\r\n", cmd);
    }
}

// ============================================================================
// Core 0 Task: Serial Communication
// ============================================================================

void serialTask(void* parameter) {
    serialPrintln("Serial task started on Core 0");

    // Debug command buffer (separate from movement command buffer)
    char debugBuffer[64];
    int debugIndex = 0;

    while (true) {
        while (Serial.available() > 0) {
            char c = Serial.read();

            if (c == 'Z') {
                // Start of movement command - takes priority
                messageStarted = true;
                bufferIndex = 0;
                serialBuffer[bufferIndex++] = c;
                debugIndex = 0;  // Cancel any debug command in progress
            }
            else if (messageStarted) {
                if (bufferIndex < MAX_MESSAGE_LENGTH - 1) {
                    serialBuffer[bufferIndex++] = c;

                    if (c == 'F') {
                        serialBuffer[bufferIndex] = '\0';

                        CommandSet cmdSet;
                        if (parseMessage(serialBuffer, &cmdSet)) {
                            if (!enqueueCommandSet(&cmdSet)) {
                                serialPrintln("E:QUEUE_FULL");
                            }
                        } else {
                            serialPrintln("E:PARSE_FAIL");
                        }

                        messageStarted = false;
                        bufferIndex = 0;
                    }
                }
                else {
                    serialPrintln("E:OVERFLOW");
                    messageStarted = false;
                    bufferIndex = 0;
                }
            }
            else if (c == '\n' || c == '\r') {
                // End of debug command
                if (debugIndex > 0) {
                    debugBuffer[debugIndex] = '\0';
                    processDebugCommand(debugBuffer);
                    debugIndex = 0;
                }
            }
            else {
                // Accumulate debug command characters
                if (debugIndex < (int)sizeof(debugBuffer) - 1) {
                    debugBuffer[debugIndex++] = c;
                }
            }
        }

        vTaskDelay(1);
    }
}

// ============================================================================
// Core 1 Task: Motor Control
// ============================================================================

void motorTask(void* parameter) {
    serialPrintln("Motor task started on Core 1");

    // Initialize PCA9685 servo driver
    initializeMotorHardware();
    serialPrintln("PCA9685 initialized, servos centered");

    serialPrintf("Servo channels: %d, %d, %d\r\n",
        SERVO_CHANNELS[0], SERVO_CHANNELS[1], SERVO_CHANNELS[2]);
    serialPrintln("Servo controller ready");
    serialPrintln("Type HELP for available commands");

    CommandSet cmdSet;

    while (true) {
        if (dequeueCommandSet(&cmdSet)) {
            if (cmdSet.isValid && cmdSet.commandCount > 0) {
                // Filter commands to valid motor indices
                CommandSet filteredCmd;
                filteredCmd.commandCount = 0;
                filteredCmd.isValid = true;

                for (int i = 0; i < cmdSet.commandCount; i++) {
                    int idx = cmdSet.commands[i].motorIndex;
                    if (idx >= 0 && idx < NUM_MOTORS) {
                        filteredCmd.commands[filteredCmd.commandCount] = cmdSet.commands[i];
                        filteredCmd.commandCount++;
                    }
                }

                if (filteredCmd.commandCount > 0) {
                    executeSynchronizedMovement(filteredCmd.commands, filteredCmd.commandCount);

                    // Send acknowledgment with positions (matching v13 format)
                    Serial.print("OK:");
                    for (int i = 0; i < filteredCmd.commandCount; i++) {
                        int idx = filteredCmd.commands[i].motorIndex;
                        serialPrintf("M%dP%d", idx, currentPositions[idx]);
                        if (i < filteredCmd.commandCount - 1) Serial.print(",");
                    }
                    serialPrintln("");
                }
            }
        }

        vTaskDelay(1);
    }
}

// ============================================================================
// Setup and Loop
// ============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
        delay(10);
    }

    serialPrintln("ESP32 Servo Controller Starting...");
    serialPrintln("Hardware: PCA9685 + SG92R Servos");

    // Create serial task on Core 0
    xTaskCreatePinnedToCore(
        serialTask,
        "SerialTask",
        4096,
        NULL,
        1,
        &serialTaskHandle,
        0
    );

    // Create motor task on Core 1
    xTaskCreatePinnedToCore(
        motorTask,
        "MotorTask",
        8192,
        NULL,
        1,
        &motorTaskHandle,
        1
    );

    serialPrintln("Tasks created. Initializing servos...");
}

void loop() {
    // All work is done in tasks
    vTaskDelay(100);
}
