/**
 * ESP32 Dual-Core DC Motor Controller with Encoder Feedback
 *
 * Hardware:
 *   - ESP32
 *   - 2x TB6612FNG motor drivers
 *   - 3x PA2-50 encoders (quadrature)
 *   - 3x 64:1 gearbox with Faulhaber 0615 DC motors
 *
 * Core 0: Receives serial commands, parses them, updates command queue
 * Core 1: Executes synchronized motor movements with PID control
 *
 * Supports 1-3 motors without reflashing (auto-detects connected motors)
 *
 * Message format: "ZM<idx>P<pos>M<idx>P<pos>...F"
 * Example: "ZM0P100M1P-50M2P75F"
 */

#include "driver/pcnt.h"
#include "driver/ledc.h"
#include "Preferences.h"  // For NVS storage of PID values

// ============================================================================
// Configuration
// ============================================================================

#define NUM_MOTORS 3
#define SERIAL_BAUD 115200
#define MAX_MESSAGE_LENGTH 128
#define COMMAND_QUEUE_SIZE 10

// Encoder configuration
#define ENCODER_PPR 50           // Pulses per revolution (PA2-50)
#define QUADRATURE_MULT 4        // x4 for full quadrature decoding
#define GEAR_RATIO 64            // 64:1 gearbox
#define COUNTS_PER_REV (ENCODER_PPR * QUADRATURE_MULT * GEAR_RATIO)  // 12800

// PWM configuration
#define VMOT_VOLTAGE 5.0f        // Your VMOT supply voltage
#define MOTOR_VOLTAGE 3.0f       // Motor's rated voltage
#define MAX_PWM_DUTY (int)((MOTOR_VOLTAGE / VMOT_VOLTAGE) * 255.0f)  // = 153 for 5V VMOT
#define PWM_FREQ 20000           // 20kHz PWM frequency
#define PWM_RESOLUTION 8         // 8-bit resolution (0-255)

// PID configuration - TUNED VALUES (reduced to prevent oscillation)
#define PID_LOOP_INTERVAL_MS 5
#define DEFAULT_KP 0.5f
#define DEFAULT_KI 0.05f
#define DEFAULT_KD 0.001f
#define PID_OUTPUT_LIMIT MAX_PWM_DUTY  
#define POSITION_TOLERANCE 10    // Encoder counts tolerance for "reached position"

// Motor detection timeout
#define MOTOR_DETECT_TIMEOUT_MS 500
#define MOTOR_DETECT_PWM 50          // PWM value for detection pulse (reduced for safety)
#define MOTOR_DETECT_DURATION_MS 30  // Duration of detection pulse in ms (reduced for safety)

// Synchronized movement configuration
#define MAX_VELOCITY 5000.0f     // Maximum velocity in encoder counts per second
#define MIN_MOVEMENT_DURATION_MS 10  // Minimum movement duration in milliseconds
#define MOVEMENT_TIMEOUT_MS 7000    // Movement timeout in milliseconds

// Motion blending
#define BLEND_ENABLED true       // Enable motion blending for continuous command streams

// ============================================================================
// GPIO Pin Definitions
// ============================================================================

// Motor 1 - TB6612FNG Board #1, Channel A
#define MOTOR1_PWM   26
#define MOTOR1_IN1   27
#define MOTOR1_IN2   14
#define MOTOR1_ENC_A 19
#define MOTOR1_ENC_B 21

// Motor 2 - TB6612FNG Board #1, Channel B
// WARNING: GPIO 12 is a strapping pin! If HIGH during boot, ESP32 fails to start.
//          Ensure TB6612FNG doesn't pull this HIGH before ESP32 boots.
//          Alternative pins: 4, 5, 23 (if available)
#define MOTOR2_PWM   13
#define MOTOR2_IN1   2
#define MOTOR2_IN2   4
#define MOTOR2_ENC_A 5
#define MOTOR2_ENC_B 18

// Motor 3 - TB6612FNG Board #2, Channel A
// WARNING: GPIO 15 is a strapping pin (affects boot debug output).
//          Less critical than GPIO 12, but may cause serial garbage during boot.
//          Alternative pins: 2, 4, 5 (if available)
#define MOTOR3_PWM   25
#define MOTOR3_IN1   33
#define MOTOR3_IN2   32
#define MOTOR3_ENC_A 16
#define MOTOR3_ENC_B 17

// Pin arrays for easy iteration
const int PWM_PINS[NUM_MOTORS] = {MOTOR1_PWM, MOTOR2_PWM, MOTOR3_PWM};
const int IN1_PINS[NUM_MOTORS] = {MOTOR1_IN1, MOTOR2_IN1, MOTOR3_IN1};
const int IN2_PINS[NUM_MOTORS] = {MOTOR1_IN2, MOTOR2_IN2, MOTOR3_IN2};
const int ENC_A_PINS[NUM_MOTORS] = {MOTOR1_ENC_A, MOTOR2_ENC_A, MOTOR3_ENC_A};
const int ENC_B_PINS[NUM_MOTORS] = {MOTOR1_ENC_B, MOTOR2_ENC_B, MOTOR3_ENC_B};

// PCNT units for each encoder
const pcnt_unit_t PCNT_UNITS[NUM_MOTORS] = {PCNT_UNIT_0, PCNT_UNIT_1, PCNT_UNIT_2};

// LEDC channels for PWM
const ledc_channel_t LEDC_CHANNELS[NUM_MOTORS] = {LEDC_CHANNEL_0, LEDC_CHANNEL_1, LEDC_CHANNEL_2};

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
void loadPIDValues();
void savePIDValues();
bool autoTunePID(int motorIdx);
void processDebugCommand(const char* cmd);

struct PIDController {
    float kp;
    float ki;
    float kd;
    float integral;
    float lastError;
    float outputLimit;
};

struct MotorState {
    volatile int32_t encoderCount;      // Current encoder position (extended beyond PCNT limit)
    volatile int32_t targetPosition;    // Target position in encoder counts
    volatile int16_t overflowCount;     // Track PCNT overflows
    PIDController pid;
    bool enabled;                        // Is this motor connected/enabled?
    bool reachedTarget;                  // Has motor reached its target?

    // Trajectory tracking for synchronized movement
    int32_t trajectoryStart;            // Start position for current trajectory
    int32_t trajectoryEnd;              // End position for current trajectory
    int32_t trajectoryDistance;         // Signed distance (end - start)

    // Stall detection
    unsigned long lastMovementTime;     // Last time encoder position changed
    int32_t lastEncoderPosition;        // Last recorded encoder position for stall detection
    int stallCount;                     // Consecutive stall detections
};

// ============================================================================
// Global Variables
// ============================================================================

MotorState motors[NUM_MOTORS];
volatile bool motorsMoving = false;
volatile int activeMotorCount = 0;

// Trajectory state for synchronized movement
struct TrajectoryState {
    unsigned long startTime;            // Movement start time in microseconds
    float duration;                     // Total movement duration in seconds
    bool active;                        // Is a trajectory currently active?
    int motorIndices[NUM_MOTORS];       // Indices of motors in this trajectory
    int motorCount;                     // Number of motors in trajectory
} trajectory;

// Command queue (protected by queueMux critical section, no volatile needed)
CommandSet commandQueue[COMMAND_QUEUE_SIZE];
int queueHead = 0;
int queueTail = 0;
portMUX_TYPE queueMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE pcntMux = portMUX_INITIALIZER_UNLOCKED;

// Serial buffer
char serialBuffer[MAX_MESSAGE_LENGTH];
int bufferIndex = 0;
bool messageStarted = false;

// Task handles
TaskHandle_t serialTaskHandle;
TaskHandle_t motorTaskHandle;

// NVS storage for PID values
Preferences preferences;

// Position scaling (from backend units to encoder counts)
const float POSITION_SCALE = COUNTS_PER_REV / 1000.0;  // 1000 backend units = 1 revolution

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
// PCNT Interrupt Handler
// ============================================================================

static void IRAM_ATTR pcntOverflowHandler(void* arg) {
    int motorIdx = (int)arg;
    uint32_t status = 0;
    pcnt_get_event_status(PCNT_UNITS[motorIdx], &status);

    portENTER_CRITICAL_ISR(&pcntMux);
    if (status & PCNT_EVT_H_LIM) {
        motors[motorIdx].overflowCount++;
    } else if (status & PCNT_EVT_L_LIM) {
        motors[motorIdx].overflowCount--;
    }
    portEXIT_CRITICAL_ISR(&pcntMux);
}


// ============================================================================
// Encoder Functions
// ============================================================================

// Flag to track if PCNT ISR service has been installed
static bool pcntIsrServiceInstalled = false;

void initEncoder(int motorIdx) {
    pcnt_config_t config = {
        .pulse_gpio_num = ENC_A_PINS[motorIdx],
        .ctrl_gpio_num = ENC_B_PINS[motorIdx],
        .lctrl_mode = PCNT_MODE_REVERSE,
        .hctrl_mode = PCNT_MODE_KEEP,
        .pos_mode = PCNT_COUNT_INC,
        .neg_mode = PCNT_COUNT_DEC,
        .counter_h_lim = 10000,
        .counter_l_lim = -10000,
        .unit = PCNT_UNITS[motorIdx],
        .channel = PCNT_CHANNEL_0,
    };
    pcnt_unit_config(&config);

    // Configure second channel for full quadrature (x4)
    config.pulse_gpio_num = ENC_B_PINS[motorIdx];
    config.ctrl_gpio_num = ENC_A_PINS[motorIdx];
    config.channel = PCNT_CHANNEL_1;
    config.pos_mode = PCNT_COUNT_DEC;
    config.neg_mode = PCNT_COUNT_INC;
    pcnt_unit_config(&config);

    // Set up overflow interrupts
    pcnt_event_enable(PCNT_UNITS[motorIdx], PCNT_EVT_H_LIM);
    pcnt_event_enable(PCNT_UNITS[motorIdx], PCNT_EVT_L_LIM);

    // Install ISR service only once (shared by all PCNT units)
    if (!pcntIsrServiceInstalled) {
        pcnt_isr_service_install(0);
        pcntIsrServiceInstalled = true;
    }
    pcnt_isr_handler_add(PCNT_UNITS[motorIdx], pcntOverflowHandler, (void*)motorIdx);

    // Enable filter to ignore glitches
    pcnt_set_filter_value(PCNT_UNITS[motorIdx], 100);
    pcnt_filter_enable(PCNT_UNITS[motorIdx]);

    // Initialize and start counter
    pcnt_counter_pause(PCNT_UNITS[motorIdx]);
    pcnt_counter_clear(PCNT_UNITS[motorIdx]);
    pcnt_counter_resume(PCNT_UNITS[motorIdx]);

    motors[motorIdx].overflowCount = 0;
}

int32_t readEncoderPosition(int motorIdx) {
    int16_t count;
    int16_t overflow;
    
    // Disable interrupts briefly to get consistent reading
    portENTER_CRITICAL_ISR(&pcntMux);
    pcnt_get_counter_value(PCNT_UNITS[motorIdx], &count);
    overflow = motors[motorIdx].overflowCount;
    portEXIT_CRITICAL_ISR(&pcntMux);
    
    return (int32_t)overflow * 10000 + count;
}

void resetEncoder(int motorIdx) {
    portENTER_CRITICAL(&pcntMux);
    pcnt_counter_pause(PCNT_UNITS[motorIdx]);
    pcnt_counter_clear(PCNT_UNITS[motorIdx]);
    motors[motorIdx].overflowCount = 0;
    motors[motorIdx].encoderCount = 0;
    pcnt_counter_resume(PCNT_UNITS[motorIdx]);
    portEXIT_CRITICAL(&pcntMux);
}

// ============================================================================
// Motor Hardware Abstraction Layer
// ============================================================================

/**
 * Initialize the motor hardware.
 * Replace this function's contents when switching motor types.
 */
void initializeMotorHardware() {
    for (int i = 0; i < NUM_MOTORS; i++) {
        // Configure direction pins
        pinMode(IN1_PINS[i], OUTPUT);
        pinMode(IN2_PINS[i], OUTPUT);
        digitalWrite(IN1_PINS[i], LOW);
        digitalWrite(IN2_PINS[i], LOW);

        // Configure PWM using LEDC
        ledc_timer_config_t timerConfig = {
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .duty_resolution = (ledc_timer_bit_t)PWM_RESOLUTION,
            .timer_num = LEDC_TIMER_0,
            .freq_hz = PWM_FREQ,
            .clk_cfg = LEDC_AUTO_CLK
        };
        ledc_timer_config(&timerConfig);

        ledc_channel_config_t channelConfig = {
            .gpio_num = PWM_PINS[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNELS[i],
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        };
        ledc_channel_config(&channelConfig);

        // Initialize encoder
        initEncoder(i);

        // Initialize PID
        motors[i].pid.kp = DEFAULT_KP;
        motors[i].pid.ki = DEFAULT_KI;
        motors[i].pid.kd = DEFAULT_KD;
        motors[i].pid.integral = 0;
        motors[i].pid.lastError = 0;
        motors[i].pid.outputLimit = PID_OUTPUT_LIMIT;

        motors[i].encoderCount = 0;
        motors[i].targetPosition = 0;
        motors[i].enabled = false;
        motors[i].reachedTarget = true;

        // Initialize trajectory tracking
        motors[i].trajectoryStart = 0;
        motors[i].trajectoryEnd = 0;
        motors[i].trajectoryDistance = 0;

        // Initialize stall detection
        motors[i].lastMovementTime = 0;
        motors[i].lastEncoderPosition = 0;
        motors[i].stallCount = 0;
    }

    // Initialize trajectory state
    trajectory.active = false;
    trajectory.motorCount = 0;
}

/**
 * Set motor speed and direction.
 * This is the low-level function that interfaces with TB6612FNG.
 *
 * @param motorIndex Index of the motor (0 to NUM_MOTORS-1)
 * @param speed Speed value (-255 to 255, negative = reverse)
 */
void setMotorSpeedRaw(int motorIndex, int speed) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return;
    if (!motors[motorIndex].enabled) return;

    speed = constrain(speed, -MAX_PWM_DUTY, MAX_PWM_DUTY);

    if (speed > 0) {
        // Forward
        digitalWrite(IN1_PINS[motorIndex], HIGH);
        digitalWrite(IN2_PINS[motorIndex], LOW);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex], speed);
    } else if (speed < 0) {
        // Reverse
        digitalWrite(IN1_PINS[motorIndex], LOW);
        digitalWrite(IN2_PINS[motorIndex], HIGH);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex], -speed);
    } else {
        // Stop (brake)
        digitalWrite(IN1_PINS[motorIndex], LOW);
        digitalWrite(IN2_PINS[motorIndex], LOW);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex], 0);
    }
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex]);
}

/**
 * Stop a motor (coast or brake).
 */
void stopMotor(int motorIndex, bool brake = true) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return;

    if (brake) {
        // Brake: both inputs HIGH
        digitalWrite(IN1_PINS[motorIndex], HIGH);
        digitalWrite(IN2_PINS[motorIndex], HIGH);
    } else {
        // Coast: both inputs LOW
        digitalWrite(IN1_PINS[motorIndex], LOW);
        digitalWrite(IN2_PINS[motorIndex], LOW);
    }
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex], 0);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIndex]);
}

/**
 * Cleanup motor hardware.
 */
void cleanupMotorHardware() {
    for (int i = 0; i < NUM_MOTORS; i++) {
        stopMotor(i);
        pcnt_isr_handler_remove(PCNT_UNITS[i]);
    }
    pcnt_isr_service_uninstall();
}

// ============================================================================
// Motor Detection
// ============================================================================

/**
 * Detect which motors are connected by applying a brief pulse and checking encoder response.
 * After detection, each motor is returned to its original position for safety.
 */
void detectConnectedMotors() {
    serialPrintln("Detecting connected motors...");
    activeMotorCount = 0;

    for (int i = 0; i < NUM_MOTORS; i++) {
        // Reset encoder
        resetEncoder(i);

        // Apply brief forward pulse (reduced movement for safety)
        digitalWrite(IN1_PINS[i], HIGH);
        digitalWrite(IN2_PINS[i], LOW);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[i], MOTOR_DETECT_PWM);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[i]);

        // DEBUG
        /*
        for (int j = 0; j < 3; j++)
        {
          serialPrintf("A=%d B=%d\r\n", digitalRead(ENC_A_PINS[i]), digitalRead(ENC_B_PINS[i]));
          delay(10);
        }
        */
        delay(MOTOR_DETECT_DURATION_MS);

        // Stop motor
        stopMotor(i);

        // Check if encoder moved
        int32_t encoderDelta = readEncoderPosition(i);

        if (abs(encoderDelta) > 5) {
            motors[i].enabled = true;
            activeMotorCount++;
            serialPrintf("Motor %d: DETECTED (encoder delta: %d)\r\n", i, abs(encoderDelta));

            // Return motor to original position (position 0)
            int32_t targetPos = 0;
            unsigned long returnStart = millis();
            const unsigned long RETURN_TIMEOUT_MS = 1000;

            while (millis() - returnStart < RETURN_TIMEOUT_MS) {
                int32_t currentPos = readEncoderPosition(i);
                int32_t error = targetPos - currentPos;

                if (abs(error) <= 3) {
                    // Close enough - stop and brake
                    stopMotor(i, true);
                    break;
                }

                // Simple proportional control to return
                int pwm = constrain(error * 2, -MOTOR_DETECT_PWM, MOTOR_DETECT_PWM);
                if (pwm > 0) {
                    digitalWrite(IN1_PINS[i], HIGH);
                    digitalWrite(IN2_PINS[i], LOW);
                } else {
                    digitalWrite(IN1_PINS[i], LOW);
                    digitalWrite(IN2_PINS[i], HIGH);
                    pwm = -pwm;
                }
                ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[i], pwm);
                ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[i]);

                delay(5);
            }

            stopMotor(i, true);
            delay(50);  // Let motor settle
        } else {
            motors[i].enabled = false;
            serialPrintf("Motor %d: NOT DETECTED\r\n", i);
        }

        // Reset encoder after detection
        resetEncoder(i);
    }

    serialPrintf("Total motors detected: %d\r\n", activeMotorCount);
}

// ============================================================================
// PID Control
// ============================================================================

float computePID(int motorIdx, float error, float dt) {
    PIDController* pid = &motors[motorIdx].pid;

    // Proportional
    float pTerm = pid->kp * error;

    // Integral with anti-windup
    pid->integral += error * dt;
    pid->integral = constrain(pid->integral, -pid->outputLimit / pid->ki, pid->outputLimit / pid->ki);
    float iTerm = pid->ki * pid->integral;

    // Derivative
    float derivative = (error - pid->lastError) / dt;
    float dTerm = pid->kd * derivative;
    pid->lastError = error;

    // Combine and limit output
    float output = pTerm + iTerm + dTerm;
    return constrain(output, -pid->outputLimit, pid->outputLimit);
}

void resetPID(int motorIdx) {
    motors[motorIdx].pid.integral = 0;
    motors[motorIdx].pid.lastError = 0;
}

// ============================================================================
// Logical Motor Control Layer
// ============================================================================

/**
 * Set target position for a motor.
 *
 * @param motorIndex Index of the motor
 * @param position Target position in backend units
 */
void setMotorTargetPosition(int motorIndex, int position) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return;
    if (!motors[motorIndex].enabled) return;

    // Convert backend position to encoder counts
    motors[motorIndex].targetPosition = (int32_t)(position * POSITION_SCALE);
    motors[motorIndex].reachedTarget = false;
    resetPID(motorIndex);
}

/**
 * Update a single motor's PID control loop.
 * Returns true if motor has reached target position.
 */
bool updateMotorPID(int motorIndex, float dt) {
    if (motorIndex < 0 || motorIndex >= NUM_MOTORS) return true;
    if (!motors[motorIndex].enabled) return true;
    if (motors[motorIndex].reachedTarget) return true;

    // Read current position
    motors[motorIndex].encoderCount = readEncoderPosition(motorIndex);

    // Calculate error
    float error = motors[motorIndex].targetPosition - motors[motorIndex].encoderCount;

    // Check if we've reached the target
    if (abs(error) <= POSITION_TOLERANCE) {
        stopMotor(motorIndex, true);
        motors[motorIndex].reachedTarget = true;
        return true;
    }

    // Compute PID output
    float output = computePID(motorIndex, error, dt);

    // Apply motor speed
    setMotorSpeedRaw(motorIndex, (int)output);

    return false;
}

/**
 * Check if there's a pending command in the queue (for motion blending).
 */
bool hasQueuedCommand() {
    portENTER_CRITICAL(&queueMux);
    bool hasCommand = (queueHead != queueTail);
    portEXIT_CRITICAL(&queueMux);
    return hasCommand;
}

/**
 * Setup a new trajectory from current positions to target positions.
 * Uses linear velocity profile - all motors complete in the same duration.
 *
 * @param commands Array of motor commands
 * @param commandCount Number of commands
 */
void setupTrajectory(MotorCommand* commands, int commandCount) {
    if (commandCount == 0) return;

    // Read current positions and calculate distances
    int32_t maxDistance = 0;
    trajectory.motorCount = 0;

    for (int i = 0; i < commandCount; i++) {
        int idx = commands[i].motorIndex;
        if (idx >= 0 && idx < NUM_MOTORS && motors[idx].enabled) {
            // Read current encoder position
            motors[idx].encoderCount = readEncoderPosition(idx);

            // Setup trajectory for this motor
            motors[idx].trajectoryStart = motors[idx].encoderCount;
            motors[idx].trajectoryEnd = (int32_t)(commands[i].targetPosition * POSITION_SCALE);
            motors[idx].trajectoryDistance = motors[idx].trajectoryEnd - motors[idx].trajectoryStart;

            // Track max distance for duration calculation
            int32_t absDistance = abs(motors[idx].trajectoryDistance);
            if (absDistance > maxDistance) {
                maxDistance = absDistance;
            }

            // Store motor index in trajectory
            trajectory.motorIndices[trajectory.motorCount++] = idx;

            // Reset PID and stall detection
            resetPID(idx);
            motors[idx].reachedTarget = false;
            motors[idx].stallCount = 0;
            motors[idx].lastMovementTime = millis();
            motors[idx].lastEncoderPosition = motors[idx].encoderCount;
        }
    }

    // Calculate movement duration based on max distance and max velocity
    // duration = distance / velocity
    if (maxDistance > 0) {
        trajectory.duration = (float)maxDistance / MAX_VELOCITY;

        // Enforce minimum duration
        float minDuration = MIN_MOVEMENT_DURATION_MS / 1000.0f;
        if (trajectory.duration < minDuration) {
            trajectory.duration = minDuration;
        }
    } else {
        trajectory.duration = MIN_MOVEMENT_DURATION_MS / 1000.0f;
    }

    trajectory.startTime = micros();
    trajectory.active = true;
}

/**
 * Execute synchronized movement for multiple motors using linear velocity profile.
 * All motors start and end at the same time by interpolating position based on
 * a single progress variable (0.0 to 1.0).
 *
 * Supports motion blending: if a new command arrives mid-movement and BLEND_ENABLED
 * is true, the trajectory is recalculated from current positions.
 *
 * @param commands Array of motor commands
 * @param commandCount Number of commands
 */
void executeSynchronizedMovement(MotorCommand* commands, int commandCount) {
    if (commandCount == 0) return;

    // Setup initial trajectory
    setupTrajectory(commands, commandCount);

    if (trajectory.motorCount == 0) return;

    motorsMoving = true;
    unsigned long lastPidUpdate = micros();
    unsigned long movementStartTime = millis();

    // Main control loop
    while (motorsMoving) {
        unsigned long now = micros();
        float dt = (now - lastPidUpdate) / 1000000.0f;

        // Check for timeout
        if (millis() - movementStartTime > MOVEMENT_TIMEOUT_MS) {
            serialPrintln("E:TIMEOUT");
            break;
        }

        // Check for motion blending - if new command available, blend to it
        if (BLEND_ENABLED && hasQueuedCommand()) {
            CommandSet newCmd;
            if (dequeueCommandSet(&newCmd)) {
                if (newCmd.isValid && newCmd.commandCount > 0) {
                    // Filter to enabled motors
                    MotorCommand filteredCommands[NUM_MOTORS];
                    int filteredCount = 0;
                    for (int i = 0; i < newCmd.commandCount; i++) {
                        int idx = newCmd.commands[i].motorIndex;
                        if (idx >= 0 && idx < NUM_MOTORS && motors[idx].enabled) {
                            filteredCommands[filteredCount++] = newCmd.commands[i];
                        }
                    }

                    if (filteredCount > 0) {
                        // Blend: setup new trajectory from current positions
                        setupTrajectory(filteredCommands, filteredCount);
                        lastPidUpdate = micros();
                        continue;
                    }
                }
            }
        }

        // Run PID at specified interval
        if (dt >= PID_LOOP_INTERVAL_MS / 1000.0f) {
            lastPidUpdate = now;

            // Calculate progress (0.0 to 1.0) based on elapsed time
            float elapsed = (now - trajectory.startTime) / 1000000.0f;
            float progress = elapsed / trajectory.duration;

            // Clamp progress to 1.0
            if (progress > 1.0f) {
                progress = 1.0f;
            }

            bool allReached = true;

            // Update each motor
            for (int i = 0; i < trajectory.motorCount; i++) {
                int idx = trajectory.motorIndices[i];

                if (motors[idx].reachedTarget) continue;

                // Read current position
                motors[idx].encoderCount = readEncoderPosition(idx);

                // Calculate interpolated target position using linear profile
                // All motors use the same progress value, ensuring synchronization
                int32_t interpolatedTarget;

                if (progress >= 1.0f) {
                    // Movement complete - target final position
                    interpolatedTarget = motors[idx].trajectoryEnd;
                } else {
                    // Linear interpolation: start + (end - start) * progress
                    interpolatedTarget = motors[idx].trajectoryStart +
                        (int32_t)(motors[idx].trajectoryDistance * progress);
                }

                // Update target position for PID
                motors[idx].targetPosition = interpolatedTarget;

                // Calculate position error (PID tracks the interpolated target)
                float error = motors[idx].targetPosition - motors[idx].encoderCount;

                // Check if motor reached final target (only when progress >= 1.0)
                if (progress >= 1.0f && abs(error) <= POSITION_TOLERANCE) {
                    stopMotor(idx, true);
                    motors[idx].reachedTarget = true;
                    continue;
                }

                // Stall detection
                if (abs(motors[idx].encoderCount - motors[idx].lastEncoderPosition) > 2) {
                    motors[idx].lastMovementTime = millis();
                    motors[idx].lastEncoderPosition = motors[idx].encoderCount;
                    motors[idx].stallCount = 0;
                } else if (millis() - motors[idx].lastMovementTime > 500) {
                    // No movement for 500ms while trying to move
                    motors[idx].stallCount++;
                    if (motors[idx].stallCount > 5) {
                        serialPrintf("W:STALL M%d\r\n", idx);
                        motors[idx].reachedTarget = true;  // Give up on this motor
                        stopMotor(idx, true);
                        continue;
                    }
                }

                // Compute and apply PID output
                float output = computePID(idx, error, dt);
                setMotorSpeedRaw(idx, (int)output);

                allReached = false;
            }

            if (allReached) {
                motorsMoving = false;
            }
        }

        taskYIELD();  // Allow other tasks to run (lower latency than vTaskDelay)
    }

    // Ensure all motors are stopped and restore PID limits
    for (int i = 0; i < trajectory.motorCount; i++) {
        int idx = trajectory.motorIndices[i];
        stopMotor(idx, true);
        motors[idx].pid.outputLimit = PID_OUTPUT_LIMIT;
    }

    trajectory.active = false;
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

        // Parse position
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
        serialPrintln("=== Motor Status ===");
        for (int i = 0; i < NUM_MOTORS; i++) {
            serialPrintf("Motor %d: %s, Pos: %d, Target: %d\r\n",
                i,
                motors[i].enabled ? "ENABLED" : "DISABLED",
                (int)(motors[i].encoderCount / POSITION_SCALE),
                (int)(motors[i].targetPosition / POSITION_SCALE));
        }
    }
    else if (strncmp(cmd, "RESET", 5) == 0) {
        for (int i = 0; i < NUM_MOTORS; i++) {
            resetEncoder(i);
            motors[i].targetPosition = 0;
        }
        serialPrintln("Encoders reset");
    }
    else if (strncmp(cmd, "DETECT", 6) == 0) {
        detectConnectedMotors();
    }
    else if (strncmp(cmd, "PID", 3) == 0) {
        // Parse PID values: PID:KP:KI:KD or PID:M:KP:KI:KD for specific motor
        float kp, ki, kd;
        int motor;
        if (sscanf(cmd, "PID:%d:%f:%f:%f", &motor, &kp, &ki, &kd) == 4) {
            // Set PID for specific motor
            if (motor >= 0 && motor < NUM_MOTORS) {
                motors[motor].pid.kp = kp;
                motors[motor].pid.ki = ki;
                motors[motor].pid.kd = kd;
                serialPrintf("Motor %d PID set to Kp=%.2f, Ki=%.2f, Kd=%.2f\r\n", motor, kp, ki, kd);
            }
        } else if (sscanf(cmd, "PID:%f:%f:%f", &kp, &ki, &kd) == 3) {
            // Set PID for all motors
            for (int i = 0; i < NUM_MOTORS; i++) {
                motors[i].pid.kp = kp;
                motors[i].pid.ki = ki;
                motors[i].pid.kd = kd;
            }
            serialPrintf("All motors PID set to Kp=%.2f, Ki=%.2f, Kd=%.2f\r\n", kp, ki, kd);
        }
    }
    else if (strncmp(cmd, "AUTOTUNE", 8) == 0) {
        // Auto-tune PID: AUTOTUNE or AUTOTUNE:M for specific motor
        int motor;
        if (sscanf(cmd, "AUTOTUNE:%d", &motor) == 1) {
            // Tune specific motor
            if (motor >= 0 && motor < NUM_MOTORS) {
                autoTunePID(motor);
            } else {
                serialPrintln("E:Invalid motor index");
            }
        } else {
            // Tune all motors
            autoTuneAllMotors();
        }
    }
    else if (strncmp(cmd, "SAVEPID", 7) == 0) {
        savePIDValues();
    }
    else if (strncmp(cmd, "LOADPID", 7) == 0) {
        loadPIDValues();
    }
    else if (strncmp(cmd, "SHOWPID", 7) == 0) {
        serialPrintln("=== PID Values ===");
        for (int i = 0; i < NUM_MOTORS; i++) {
            serialPrintf("Motor %d: Kp=%.3f Ki=%.3f Kd=%.3f\r\n",
                i, motors[i].pid.kp, motors[i].pid.ki, motors[i].pid.kd);
        }
    }
    else if (strncmp(cmd, "HELP", 4) == 0) {
        serialPrintln("=== Commands ===");
        serialPrintln("STATUS     - Show motor status");
        serialPrintln("RESET      - Reset encoder positions");
        serialPrintln("DETECT     - Re-detect motors");
        serialPrintln("PID:Kp:Ki:Kd        - Set PID for all motors");
        serialPrintln("PID:M:Kp:Ki:Kd      - Set PID for motor M");
        serialPrintln("AUTOTUNE   - Auto-tune all motors");
        serialPrintln("AUTOTUNE:M - Auto-tune motor M");
        serialPrintln("SAVEPID    - Save PID to flash");
        serialPrintln("LOADPID    - Load PID from flash");
        serialPrintln("SHOWPID    - Show current PID values");
        serialPrintln("ZM<n>P<pos>F - Move motor n to position");
    }
    else {
        serialPrintf("Unknown command: %s\r\n", cmd);
    }
}

// ============================================================================
// Core 0 Task: Serial Communication (handles ALL serial input)
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
                // Continue accumulating movement command
                if (bufferIndex < MAX_MESSAGE_LENGTH - 1) {
                    serialBuffer[bufferIndex++] = c;

                    if (c == 'F') {
                        // Movement command complete
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

    // Initialize motor hardware
    initializeMotorHardware();

    // Detect connected motors
    detectConnectedMotors();

    if (activeMotorCount == 0) {
        serialPrintln("WARNING: No motors detected! Check connections.");
    }

    // Load saved PID values from NVS (if any)
    loadPIDValues();

    serialPrintln("Motor controller ready");
    serialPrintln("Type HELP for available commands");

    CommandSet cmdSet;

    while (true) {
        if (dequeueCommandSet(&cmdSet)) {
            if (cmdSet.isValid && cmdSet.commandCount > 0) {
                // Filter commands to only include enabled motors
                CommandSet filteredCmd;
                filteredCmd.commandCount = 0;
                filteredCmd.isValid = true;

                for (int i = 0; i < cmdSet.commandCount; i++) {
                    int idx = cmdSet.commands[i].motorIndex;
                    if (idx >= 0 && idx < NUM_MOTORS && motors[idx].enabled) {
                        filteredCmd.commands[filteredCmd.commandCount] = cmdSet.commands[i];
                        filteredCmd.commandCount++;
                    }
                }

                if (filteredCmd.commandCount > 0) {
                    executeSynchronizedMovement(filteredCmd.commands, filteredCmd.commandCount);

                    // Send acknowledgment with positions
                    Serial.print("OK:");
                    for (int i = 0; i < filteredCmd.commandCount; i++) {
                        int idx = filteredCmd.commands[i].motorIndex;
                        serialPrintf("M%dP%d", idx, (int)(motors[idx].encoderCount / POSITION_SCALE));
                        if (i < filteredCmd.commandCount - 1) Serial.print(",");
                    }
                    serialPrintln("");
                }
            }
        }

        vTaskDelay(1);  // Reduced from 5ms to 1ms for lower latency
    }
}

// ============================================================================
// PID Storage (NVS)
// ============================================================================

void savePIDValues() {
    preferences.begin("motor_pid", false);
    for (int i = 0; i < NUM_MOTORS; i++) {
        char key[16];
        snprintf(key, sizeof(key), "m%d_kp", i);
        preferences.putFloat(key, motors[i].pid.kp);
        snprintf(key, sizeof(key), "m%d_ki", i);
        preferences.putFloat(key, motors[i].pid.ki);
        snprintf(key, sizeof(key), "m%d_kd", i);
        preferences.putFloat(key, motors[i].pid.kd);
    }
    preferences.end();
    serialPrintln("PID values saved to NVS");
}

void loadPIDValues() {
    preferences.begin("motor_pid", true);
    for (int i = 0; i < NUM_MOTORS; i++) {
        char key[16];
        snprintf(key, sizeof(key), "m%d_kp", i);
        if (preferences.isKey(key)) {
            motors[i].pid.kp = preferences.getFloat(key, DEFAULT_KP);
            snprintf(key, sizeof(key), "m%d_ki", i);
            motors[i].pid.ki = preferences.getFloat(key, DEFAULT_KI);
            snprintf(key, sizeof(key), "m%d_kd", i);
            motors[i].pid.kd = preferences.getFloat(key, DEFAULT_KD);
            serialPrintf("Loaded PID for Motor %d: Kp=%.3f Ki=%.3f Kd=%.3f\r\n",
                i, motors[i].pid.kp, motors[i].pid.ki, motors[i].pid.kd);
        }
    }
    preferences.end();
}

// ============================================================================
// PID Auto-Tuning (Relay Feedback / Åström-Hägglund Method)
// ============================================================================

/**
 * Auto-tune PID for a single motor using relay feedback method.
 * The motor oscillates around a setpoint and we measure the oscillation
 * to calculate optimal PID values.
 *
 * @param motorIdx Index of the motor to tune
 * @return true if tuning was successful
 */
bool autoTunePID(int motorIdx) {
    if (motorIdx < 0 || motorIdx >= NUM_MOTORS) return false;
    if (!motors[motorIdx].enabled) {
        serialPrintf("E:Motor %d not enabled\r\n", motorIdx);
        return false;
    }

    serialPrintf("Starting auto-tune for Motor %d...\r\n", motorIdx);

    // Configuration
    const float relayAmplitude = 150;  // PWM value for relay output
    const int tuneDistance = 500;      // Distance to move for tuning (encoder counts)
    const int maxCycles = 10;          // Number of oscillation cycles to measure
    const unsigned long maxTuneTime = 30000;  // Maximum tuning time (30 seconds)

    // Get starting position and set setpoint
    int32_t startPos = readEncoderPosition(motorIdx);
    int32_t setpoint = startPos + tuneDistance;

    // Arrays to store oscillation data
    float peaks[maxCycles];
    float valleys[maxCycles];
    unsigned long peakTimes[maxCycles];
    unsigned long valleyTimes[maxCycles];
    int peakCount = 0;
    int valleyCount = 0;

    // State tracking
    bool wasAboveSetpoint = false;
    int32_t lastPos = startPos;
    int32_t localMax = startPos;
    int32_t localMin = startPos;
    unsigned long startTime = millis();

    serialPrintln("Tuning in progress... (motor will oscillate)");

    // Run relay control until we have enough oscillation data
    while ((peakCount < maxCycles || valleyCount < maxCycles) &&
           (millis() - startTime < maxTuneTime)) {

        int32_t pos = readEncoderPosition(motorIdx);
        float error = setpoint - pos;

        // Track local extrema
        if (pos > localMax) localMax = pos;
        if (pos < localMin) localMin = pos;

        // Detect setpoint crossings and record extrema
        bool isAboveSetpoint = (pos >= setpoint);

        if (isAboveSetpoint && !wasAboveSetpoint) {
            // Crossed from below to above - record valley (minimum)
            if (valleyCount < maxCycles) {
                valleys[valleyCount] = localMin;
                valleyTimes[valleyCount] = millis();
                valleyCount++;
            }
            localMin = pos;  // Reset for next cycle
        }
        else if (!isAboveSetpoint && wasAboveSetpoint) {
            // Crossed from above to below - record peak (maximum)
            if (peakCount < maxCycles) {
                peaks[peakCount] = localMax;
                peakTimes[peakCount] = millis();
                peakCount++;
            }
            localMax = pos;  // Reset for next cycle
        }

        wasAboveSetpoint = isAboveSetpoint;
        lastPos = pos;

        // Apply relay (bang-bang) control
        if (error > 0) {
            // Below setpoint - drive forward
            digitalWrite(IN1_PINS[motorIdx], HIGH);
            digitalWrite(IN2_PINS[motorIdx], LOW);
            ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIdx], (int)relayAmplitude);
        } else {
            // Above setpoint - drive reverse
            digitalWrite(IN1_PINS[motorIdx], LOW);
            digitalWrite(IN2_PINS[motorIdx], HIGH);
            ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIdx], (int)relayAmplitude);
        }
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNELS[motorIdx]);

        delayMicroseconds(500);  // Small delay for stable readings
    }

    // Stop motor
    stopMotor(motorIdx, true);

    // Check if we got enough data
    if (peakCount < 3 || valleyCount < 3) {
        serialPrintln("E:AUTOTUNE_FAIL - Not enough oscillation cycles");
        return false;
    }

    // Calculate average period and amplitude (skip first cycle as it may be incomplete)
    float avgPeriod = 0;
    float avgAmplitude = 0;
    int validCycles = 0;

    for (int i = 1; i < min(peakCount, valleyCount); i++) {
        // Period is time between consecutive peaks (full cycle)
        if (i < peakCount - 1) {
            avgPeriod += (peakTimes[i + 1] - peakTimes[i]);
            validCycles++;
        }
        // Amplitude is peak-to-peak / 2
        avgAmplitude += abs(peaks[i] - valleys[i]);
    }

    if (validCycles == 0) {
        serialPrintln("E:AUTOTUNE_FAIL - Could not calculate period");
        return false;
    }

    avgPeriod /= validCycles;
    avgAmplitude /= (min(peakCount, valleyCount) - 1);

    serialPrintf("Measured: Period=%.1fms, Amplitude=%.1f counts\r\n", avgPeriod, avgAmplitude);

    // Calculate ultimate gain (Ku) and ultimate period (Tu)
    // Ku = 4d / (π * a) where d = relay amplitude, a = oscillation amplitude
    float Ku = (4.0f * relayAmplitude) / (3.14159f * avgAmplitude);
    float Tu = avgPeriod / 1000.0f;  // Convert to seconds

    // Conservative Ziegler-Nichols PID tuning formulas (reduced from standard to prevent oscillation)
    float newKp = 0.33f * Ku;    // Standard is 0.6
    float newKi = 0.66f * Ku / Tu;  // Standard is 1.2
    float newKd = 0.08f * Ku * Tu;  // Standard is 0.075

    // Apply some safety limits
    newKp = constrain(newKp, 0.1f, 10.0f);
    newKi = constrain(newKi, 0.01f, 2.0f);
    newKd = constrain(newKd, 0.0001f, 0.5f);

    // Apply new PID values
    motors[motorIdx].pid.kp = newKp;
    motors[motorIdx].pid.ki = newKi;
    motors[motorIdx].pid.kd = newKd;
    motors[motorIdx].pid.integral = 0;
    motors[motorIdx].pid.lastError = 0;

    serialPrintf("Auto-tune Motor %d complete: Kp=%.3f Ki=%.3f Kd=%.3f\r\n",
        motorIdx, newKp, newKi, newKd);

    // Return motor to starting position
    serialPrintln("Returning to start position...");
    motors[motorIdx].trajectoryStart = readEncoderPosition(motorIdx);
    motors[motorIdx].trajectoryEnd = startPos;
    motors[motorIdx].trajectoryDistance = motors[motorIdx].trajectoryEnd - motors[motorIdx].trajectoryStart;
    motors[motorIdx].reachedTarget = false;

    unsigned long returnStart = millis();
    while (!motors[motorIdx].reachedTarget && (millis() - returnStart < 5000)) {
        motors[motorIdx].encoderCount = readEncoderPosition(motorIdx);
        float error = startPos - motors[motorIdx].encoderCount;

        if (abs(error) <= POSITION_TOLERANCE) {
            motors[motorIdx].reachedTarget = true;
            break;
        }

        float output = computePID(motorIdx, error, 0.005f);
        setMotorSpeedRaw(motorIdx, (int)output);
        delay(5);
    }
    stopMotor(motorIdx, true);

    return true;
}

/**
 * Auto-tune all enabled motors sequentially.
 */
void autoTuneAllMotors() {
    serialPrintln("=== Auto-tuning all motors ===");
    for (int i = 0; i < NUM_MOTORS; i++) {
        if (motors[i].enabled) {
            autoTunePID(i);
            delay(500);  // Brief pause between motors
        }
    }
    serialPrintln("=== Auto-tune complete ===");
}

// ============================================================================
// Setup and Loop
// ============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
        delay(10);
    }

    serialPrintln("ESP32 DC Motor Controller Starting...");
    serialPrintln("Hardware: TB6612FNG + PA2-50 Encoders + 64:1 Gearbox");

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

    serialPrintln("Tasks created. Waiting for motor detection...");
}

void loop() {
    // All serial handling is now in serialTask
    // loop() is essentially idle
    vTaskDelay(100);
}