"""
This contains functionality pertaining to the general operation of the car
"""
from Pi_GPIO_Class import Pi_GPIO


class Pi_Car:
    def __init__(self):
        self.RPi_GPIO = Pi_GPIO()
        self.RPi_GPIO.reset_GPIO()
        self.speed = 50
        self.RPi_GPIO.set_speed(self.speed)
        self.GPS_Coords = [0, 0, 0]

    def exit(self):
        self.RPi_GPIO.reset_GPIO()

    def forward(self):
        self.RPi_GPIO.forward()
    
    def reverse(self):
        self.RPi_GPIO.reverse()

    def rotate_left(self):
        self.RPi_GPIO.rotate_left()

    def rotate_right(self):
        self.RPi_GPIO.rotate_right()

    def update_speed(self, speed):
        self.speed = speed
        self.RPi_GPIO.set_speed(self.speed)

    def stop(self):
        self.RPi_GPIO.stop()


def car_test():
    car = Pi_Car()
    running = True

    while running:
        command = input("Enter Operation: ")
        if command == "w":
            car.forward()
        elif command == "s":
            car.reverse()
        elif command == "a":
            car.rotate_left()
        elif command == "d":
            car.rotate_right()
        elif command == "m":
            speed = int(input("Enter Speed: "))
            car.update_speed(speed)
        elif command == "q":
            car.exit()


if __name__ == "main":
    car = Pi_Car()
