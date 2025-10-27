import typer
import socket

from backend import MotorMovement

def build_message(motors: list[MotorMovement]) -> str:
    message = "Z"
    for motor in motors:
        message += f"M{motor.index}P{motor.pos}"
    message += "F"
    return message

def main(pos: int = 100, motor_count: int = 3, inc: int = 0):
    motors = []
    
    for i in range(motor_count):
        motors.append(MotorMovement(pos=pos + inc * i, index=i))

    # Build message
    message = build_message(motors)

    technosoft_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    technosoft_socket.sendto(message.encode("utf-8"), ("localhost", 12347))
    technosoft_socket.close()

    print("Message sent:", message)

if __name__ == "__main__":
    typer.run(main)
