
import RPi.GPIO as GPIO
import time

led_pin = 21 
GPIO.setmode(GPIO.BCM)

GPIO.setup(led_pin, GPIO.OUT)

try:
	print("Enciendiendo LED")
	GPIO.output(led_pin, GPIO.HIGH)
	time.sleep(40)
	print("Apagando LED")
	GPIO.output(led_pin, GPIO.LOW)
except KeyboardInterrupt:
	print("Programa interrumpido por usuario")
finally:
	GPIO.cleanup()
