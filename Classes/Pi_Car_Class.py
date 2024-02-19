"""
This contains functionality pertaining to the general operation of the car
"""
from Pi_GPIO_Class import Pi_GPIO
from CV_Camera_Distance import calibrate_camera, reconstruction, reconstruction_multi, take_calibration_images, take_mapping_image
import os
import glob


class Pi_Car:
    def __init__(self):
        print("Init Car")
        self.RPi_GPIO = Pi_GPIO()
        self.RPi_GPIO.reset_GPIO()
        self.speed = 100
        self.RPi_GPIO.set_speed(self.speed)
        self.GPS_Coords = [0, 0, 0]
        self.map = None

        print("Beginning Mapping")
        # Initialize Mapping
        if not os.path.exists("Calibration Images/camera_calibration.pickle"):
            calibrate_camera()
        else:
            print("Creating initial map")
            take_mapping_image()
            self.forward()
            self.RPi_GPIO.forward()
            take_mapping_image()
            images = glob.glob("Mapping Images/*.jpg")
            self.map = reconstruction_multi(images)

    def calibrate_camera(self):
        take_calibration_images(12, 5)
        calibrate_camera()

    def plan_path():
        pass

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


if __name__ == "__main__":
    car = Pi_Car()
