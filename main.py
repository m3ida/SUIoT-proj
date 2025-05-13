import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG = 23
ECHO = 24

# Set up GPIO pins
# GPIO.setup(TRIG, GPIO.OUT)
# GPIO.setup(ECHO, GPIO.IN)

# def get_distance():
#     # Ensure trigger is low
#     print(0)
#     GPIO.output(TRIG, False)
#     print(1)

#     time.sleep(0.05)
#     print(2)

#     # Send a 10us pulse to trigger
#     GPIO.output(TRIG, True)
#     time.sleep(0.00001)  # 10us
#     GPIO.output(TRIG, False)

#     print(3)

#     pulse_start = time.time()
#     # Wait for echo to go high
#     while GPIO.input(ECHO) == 0:
#         pulse_start = time.time()

#     print(4)

#     pulse_end = time.time()
#     # Wait for echo to go low
#     while GPIO.input(ECHO) == 1:
#         pulse_end = time.time()

#     print(5)


#     # Calculate pulse duration
#     pulse_duration = pulse_end - pulse_start

#     # Distance calculation: Speed of sound = 34300 cm/s
#     distance = pulse_duration * 17150
#     distance = round(distance, 2)

#     return distance

try:
    while True:
        print(GPIO.input(ECHO))
        try:
            GPIO.output(TRIG, False)
            time.sleep(1)
            GPIO.output(TRIG, True)
            time.sleep(1)
            # dist = get_distance()
            # print(f"Distance: {dist} cm")
        except TimeoutError as e:
            print(f"Measurement error: {e}")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
    GPIO.cleanup()


