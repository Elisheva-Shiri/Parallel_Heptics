#include <Servo.h>

Servo myservo;  // create servo object to control a servo
const int STOP_SPEED = 90;    // Value to stop the servo (usually 90)
const int SPEED = 20;         // Speed value to rotate (adjust as needed)
const int RIGHT_SPEED = STOP_SPEED + SPEED;  // Speed for right rotation
const int LEFT_SPEED = STOP_SPEED - SPEED;   // Speed for left rotation
const int MOVE_TIME = 100;    // Time in milliseconds for each movement (adjust as needed)

// Keep track of total rotation to return to start position
int totalRotation = 0;  // Positive for right turns, negative for left turns

void setup() {
  Serial.begin(115200);       // initialize serial communication at 115200 baud
  myservo.attach(9);         // attaches the servo on pin 9 to the servo object
  myservo.write(STOP_SPEED); // Initially stop the servo
  Serial.println("360 Servo Control Ready");
} 

void loop() {
  while (Serial.available() > 0) {  // check if data is available to read
    char command = Serial.read(); // read the incoming byte
    
    // Print received command for debugging
    Serial.print("Received command: ");
    Serial.println(command);

    switch (command) {
      case 'S':
        // whatever
        break;

      case 'I':
      case 'M':
      case 'R':
        char direction = Serial.read(); // read the incoming byte
        
    }

    switch(command) {
      // ! Missing support for Left and Right ('L', 'R')
      // ! Missing support for stiffness value (integer)

      case 'U':  // short turn right
        myservo.write(RIGHT_SPEED);
        delay(MOVE_TIME);
        myservo.write(STOP_SPEED);
        totalRotation += 1;  // Keep track of right turn
        Serial.println("Short turn right");
        break;
        
      case 'D':  // short turn left
        myservo.write(LEFT_SPEED);
        delay(MOVE_TIME);
        myservo.write(STOP_SPEED);
        totalRotation -= 1;  // Keep track of left turn
        Serial.println("Short turn left");
        break;

      case 'S':  // return to original position
        if (totalRotation > 0) {
          // Need to turn left to return to start
          myservo.write(LEFT_SPEED);
          delay(MOVE_TIME * totalRotation);
          myservo.write(STOP_SPEED);
        } else if (totalRotation < 0) {
          // Need to turn right to return to start
          myservo.write(RIGHT_SPEED);
          delay(MOVE_TIME * -totalRotation);
          myservo.write(STOP_SPEED);
        }
        totalRotation = 0;  // Reset rotation counter
        Serial.println("Returned to start position");
        break;

      default:
        // Ignore any other characters
        myservo.write(STOP_SPEED);
        break;
    }
  }

  // ? Do we clear the buffer?
  // // Clear any remaining characters in the buffer
  // while(Serial.available() > 0) {
  //   Serial.read();
  // }
}