import time
import board
import digitalio

pwm1 = digitalio.DigitalInOut(board.D18)
pwm1.direction = digitalio.Direction.OUTPUT
pwm2 = digitalio.DigitalInOut(board.D23)
pwm2.direction = digitalio.Direction.OUTPUT

def forward():
	pwm1.value = 1
	pwm2.value = 0
	time.sleep(0.1)

def stop():
	pwm1.value = 0
	pwm2.value = 0
	time.sleep(0.2)

