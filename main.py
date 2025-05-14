import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG = 13
ECHO = 19

# Set up GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

time.sleep(2)

decoder = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

list_license_plates= []
cropped_licenses = []
model = YOLO("./license_detection.pt")
modelDigits = YOLO("./digit_detection_font4.pt")
# # modelClassifyDigits = YOLO("./digit_classification_X.pt")
classification_model = tflite.Interpreter(model_path="model.tflite")
classification_model.allocate_tensors()
input_details = classification_model.get_input_details()
output_details = classification_model.get_output_details()

def classify_character_tflite(image):
    resized_img = resize_with_padding(image)
    input_arr = np.asarray(resized_img).astype(np.float32)

    # Normalize if model requires (assumes [0,1] input)
    input_arr = input_arr / 255.0

    input_tensor = np.expand_dims(input_arr, axis=0)

    classification_model.set_tensor(input_details[0]['index'], input_tensor)
    classification_model.invoke()

    output_data = classification_model.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    return decoder[prediction]

def debug_imshow(title, image, waitKey=False):
    #cv2.imshow(title, image)

    if waitKey:
        # cv2.waitKey(0)
        time.sleep(2)

def resize_with_padding(image):
    image = Image.fromarray(image)
    width, height = image.size

    new_size = (width * 10, height * 10)
    image = image.resize(new_size, Image.LANCZOS)

    target_size = (80, 60)
    fill_color = (255, 255, 255)

    image.thumbnail(target_size, Image.LANCZOS)

    new_im = Image.new("RGB", target_size, fill_color)

    x_offset = (target_size[0] - image.size[0]) // 2
    y_offset = (target_size[1] - image.size[1]) // 2

    new_im.paste(image, (x_offset, y_offset))

    return new_im

def analyse_image(image):
    results = model([image])
    if(len(results) == 0):
        print("No License Plate found")
    for id2,result in enumerate(results):
        boxes = result.boxes

        if(len(boxes.xywh)>0):
            xywh_cpu = boxes.xywh.to('cpu').numpy()

            x = xywh_cpu[0][0]
            y = xywh_cpu[0][1]
            w = xywh_cpu[0][2]
            h = xywh_cpu[0][3]

            license_plate_crop = result.orig_img[int(y-h/2-5):int(y+h/2+5), int(x-w/2-5):int(x+w/2+5)]

            resultsDigits = modelDigits([license_plate_crop])

            license_lennet = ""
            # license_yolo = ""

            for id,r in enumerate(resultsDigits):
                boxes = r.boxes

                if(len(boxes.xywh)>0):
                    xywh_cpu = boxes.xywh.to('cpu').numpy()

                    print(xywh_cpu)

                sorted_boxes = sorted(xywh_cpu, key=lambda x: x[0])

                heights = [coords[3] for coords in sorted_boxes]
                avg_height = np.mean(heights)

                img_h, img_w = license_plate_crop.shape[:2]

                filtered_boxes = [
                coords for coords in sorted_boxes
                if coords[3] > 0.5 * avg_height
                and 0.3 < coords[1] / img_h < 0.7
                and 0.05 < coords[0] / img_w < 0.85
            ]


                for coords in filtered_boxes:
                    x = coords[0]
                    y = coords[1]
                    w = coords[2]
                    h = coords[3]

                    cropped_character = r.orig_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

                    # class_res = modelClassifyDigits(cropped_character)
                    # print("Prediction YOLO", decoder[int(class_res[0].probs.top1)])
                    # license_yolo += decoder[int(class_res[0].probs.top1)]

                    prediction_lennet = classify_character_tflite(cropped_character)
                    print("Prediction LENNET:", prediction_lennet)
                    license_lennet += prediction_lennet

            print("Prediction LENNET:", license_lennet)
            # print("Prediction YOLO:", license_yolo)

            # debug_imshow("License Plate", license_plate_crop, waitKey=True)
        else:
            print("No License Plate found")


def get_distance():
    # Ensure trigger is low
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    # Send a 10us pulse to trigger
    GPIO.output(TRIG, True)
    time.sleep(0.00001)  # 10us
    GPIO.output(TRIG, False)

    # Wait for echo to go high
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    # Wait for echo to go low
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Calculate pulse duration
    pulse_duration = pulse_end - pulse_start

    # Distance calculation: Speed of sound = 34300 cm/s
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

try:
    print("Initialized")
    while True:
        dist = get_distance()
        print(f"Distance: {dist} cm")

        if(dist < 1500):
            image = picam2.capture_array()
            picam2.capture_file("test.jpg")
            print("image captured")
            time.sleep(10)

            print("waiting 10s before checking again")
            analyse_image(image)

        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
    GPIO.cleanup()
