"""
This class contains all the PI GPIO finctionality and resets the GPIO pins on
startup and exit.
"""
import RPi.GPIO as GPIO


class Pi_GPIO:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        # Front Motor Controller Pins
        self.FMCB1_LWR = 16
        self.FMCB2_LWF = 18
        self.FMCB3_RWR = 13
        self.FMCB4_RWF = 11
        self.FMCBA_PCM = 33  # Left Motor Speed Pin
        self.FMCBB_PCM = 35  # Right Motor Speed Pin

        # Rear Motor Controller Pins
        self.RMCB1_LWR = 22
        self.RMCB2_LWF = 29
        self.RMCB3_RWR = 31
        self.RMCB4_RWF = 15
        self.RMCBA_PCM = 12  # Left Motor Speed Pin
        self.RMCBB_PCM = 32  # Right Motor Speed Pin

        # Font Wheels
        GPIO.setup(self.FMCB1_LWR, GPIO.OUT)
        GPIO.setup(self.FMCB2_LWF, GPIO.OUT)
        GPIO.setup(self.FMCB3_RWR, GPIO.OUT)
        GPIO.setup(self.FMCB4_RWF, GPIO.OUT)

        # Rear Wheels
        GPIO.setup(self.RMCB1_LWR, GPIO.OUT)
        GPIO.setup(self.RMCB2_LWF, GPIO.OUT)
        GPIO.setup(self.RMCB3_RWR, GPIO.OUT)
        GPIO.setup(self.RMCB4_RWF, GPIO.OUT)

        # Font Wheels Speed
        GPIO.setup(self.FMCBA_PCM, GPIO.OUT)
        GPIO.setup(self.FMCBB_PCM, GPIO.OUT)
        self.pwm_fl = GPIO.PWM(self.FMCBA_PCM, 1000)
        self.pwm_fr = GPIO.PWM(self.FMCBB_PCM, 1500)

        # Rear Wheels Speed
        GPIO.setup(self.RMCBA_PCM, GPIO.OUT)
        GPIO.setup(self.RMCBB_PCM, GPIO.OUT)
        self.pwm_bl = GPIO.PWM(self.RMCBA_PCM, 2000)
        self.pwm_br = GPIO.PWM(self.RMCBB_PCM, 2500)

        # Initialize Speed as 0
        self.pwm_fl.start(0)
        self.pwm_fr.start(0)
        self.pwm_bl.start(0)
        self.pwm_br.start(0)

    def reset_GPIO(self):
        self.pwm_fl.stop()
        self.pwm_fr.stop()
        self.pwm_bl.stop()
        self.pwm_br.stop()
        GPIO.cleanup()
    
    def set_speed(self, speed):
        self.pwm_fl.ChangeDutyCycle(speed)
        self.pwm_fr.ChangeDutyCycle(speed)
        self.pwm_bl.ChangeDutyCycle(speed)
        self.pwm_br.ChangeDutyCycle(speed)

    def forward(self):
        GPIO.output(self.FMCB2_LWF, GPIO.HIGH)
        GPIO.output(self.FMCB4_RWF, GPIO.HIGH)
        GPIO.output(self.RMCB2_LWF, GPIO.HIGH)
        GPIO.output(self.RMCB4_RWF, GPIO.HIGH)

    def reverse(self):
        GPIO.output(self.FMCB1_LWR, GPIO.HIGH)
        GPIO.output(self.FMCB3_RWR, GPIO.HIGH)
        GPIO.output(self.RMCB1_LWR, GPIO.HIGH)
        GPIO.output(self.RMCB3_RWR, GPIO.HIGH)

    def rotate_left(self):
        GPIO.output(self.FMCB1_LWR, GPIO.HIGH)
        GPIO.output(self.FMCB4_RWF, GPIO.HIGH)
        GPIO.output(self.RMCB1_LWR, GPIO.HIGH)
        GPIO.output(self.RMCB4_RWF, GPIO.HIGH)

    def rotate_right(self):
        GPIO.output(self.FMCB2_LWF, GPIO.HIGH)
        GPIO.output(self.FMCB3_RWR, GPIO.HIGH)
        GPIO.output(self.RMCB2_LWF, GPIO.HIGH)
        GPIO.output(self.RMCB3_RWR, GPIO.HIGH)

    def stop(self):
        GPIO.output(self.FMCB1_LWR, GPIO.LOW)
        GPIO.output(self.FMCB3_RWR, GPIO.LOW)
        GPIO.output(self.RMCB1_LWR, GPIO.LOW)
        GPIO.output(self.RMCB3_RWR, GPIO.LOW)
        GPIO.output(self.FMCB2_LWF, GPIO.LOW)
        GPIO.output(self.FMCB4_RWF, GPIO.LOW)
        GPIO.output(self.RMCB2_LWF, GPIO.LOW)
        GPIO.output(self.RMCB4_RWF, GPIO.LOW)
