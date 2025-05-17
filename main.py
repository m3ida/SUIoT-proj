import RPi.GPIO as GPIO
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Set GPIO mode
GPIO.setmode(GPIO.BOARD)

# Define GPIO pins
TRIG = 31
ECHO = 33

# Set up GPIO pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

decoder = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

list_license_plates= []
cropped_licenses = []
model = YOLO("./license_detection.pt").to(device)
modelDigits = YOLO("./digit_detection_font4.pt").to(device)
modelClassifyDigits = YOLO("./digit_classification_X.pt").to(device)
classification_model = tf.saved_model.load("./best_model")

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def classify_character(image):
    resized_img = resize_with_padding(image)
    input_arr = tf.keras.utils.img_to_array(resized_img)
    img_tensor = np.expand_dims(input_arr, axis=0)

    imageClassified = classification_model(img_tensor)
    predictions = np.argmax(imageClassified, axis=1)

    return decoder[predictions[0]]

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

            #im = Image.fromarray(license_plate_crop)
            #im.save("license_plate_crop.jpeg")

            resultsDigits = modelDigits([license_plate_crop])

            license_lennet = ""
            license_yolo = ""

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

                    class_res = modelClassifyDigits(cropped_character)
                    print("Prediction YOLO", decoder[int(class_res[0].probs.top1)])
                    license_yolo += decoder[int(class_res[0].probs.top1)]

                    prediction_lennet = classify_character(cropped_character)
                    print("Prediction LENNET:", prediction_lennet)
                    license_lennet += prediction_lennet

            print("Prediction LENNET:", license_lennet)
            print("Prediction License YOLO:", license_yolo)

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
    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    # Wait for echo to go low
    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Calculate pulse duration
    pulse_duration = pulse_end - pulse_start

    # Distance calculation: Speed of sound = 34300 cm/s
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

print(gstreamer_pipeline(flip_method=2, framerate=5, capture_width=3280*.66, capture_height=2464*.66, display_width=3280*.66, display_height=2464*.66))
cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2, capture_width=1280, capture_height=960), cv2.CAP_GSTREAMER)
gamma=3
inv_gama= 1.0/gamma

if cam.isOpened():
    try:
        print("Initialized")
        while True:
            dist = get_distance()
            print(f"Distance: {dist} cm")

            if(dist < 1500): 
               
                result, image = cam.read()
                print(result)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                gamma_corrected = np.array(255*((image_rgb /255) ** inv_gama), dtype="uint8")
                #image_rgb[:, :, 0] = image_rgb[:, :, 0] * 0.6  # Reduce Red channel
                #image_rgb[:, :, 1] = image_rgb[:, :, 1] * 0.6  # Reduce Green channel
                #image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1, beta=50)

                #hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
                #hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)
                #saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

                
                im = Image.fromarray(gamma_corrected)
                im.save("test.jpg")
                print("image captured")
                time.sleep(10)

                print("waiting 10s before checking again")
                analyse_image(gamma_corrected)
            time.sleep(1)
    finally:
        cam.release()
else:
    print("Error: Unable to open camera")
        


print("Measurement stopped by user")
cam.release()
GPIO.cleanup()
