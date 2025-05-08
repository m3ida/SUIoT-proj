from ultralytics import YOLO
from imutils import paths
import cv2
from tqdm import tqdm
# import imutils
import tensorflow as tf
import numpy as np
import os
# from fast_plate_ocr import ONNXPlateRecognizer
from PIL import Image, ImageDraw
import pytesseract

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

imagePaths = sorted(list(paths.list_images("./license_plates/group1")))
decoder = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

list_license_plates= []
cropped_licenses = []
model = YOLO("./license_detection.pt")  # pretrained YOLO11n model
modelDigits = YOLO("./digit_detection_font4.pt")  # pretrained YOLO11n model
modelClassifyDigits = YOLO("./digit_classification_X.pt")  # pretrained YOLO11n model

classification_model = tf.saved_model.load("./best_model")

def debug_imshow(title, image, waitKey=False):
    cv2.imshow(title, image)

    if waitKey:
        cv2.waitKey(0)

def read_license_plate(plate_img):
    _, thresh = cv2.threshold(plate_img, 150, 255, cv2.THRESH_BINARY)

    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)

    text = ''.join(filter(str.isalnum, text))

    if len(text) == 6:
        return text
    else:
        return text[:6] if text else "No text detected"

def resize_with_padding(image):
    image = Image.fromarray(image)
    width, height = image.size

    new_size = (width * 10, height * 10)
    image = image.resize(new_size, Image.LANCZOS)

    target_size = (100, 75)
    fill_color = (255, 255, 255)

    image.thumbnail(target_size, Image.LANCZOS)

    new_im = Image.new("RGB", target_size, fill_color)

    x_offset = (target_size[0] - image.size[0]) // 2
    y_offset = (target_size[1] - image.size[1]) // 2

    new_im.paste(image, (x_offset, y_offset))

    return new_im

for id,imagePath in enumerate(tqdm(imagePaths)):

    print(imagePath)
    results = model([imagePath])
    for id2,result in enumerate(results):
        boxes = result.boxes

        if(len(boxes.xywh)>0):
            xywh_cpu = boxes.xywh.to('cpu').numpy()

            x = xywh_cpu[0][0]
            y = xywh_cpu[0][1]
            w = xywh_cpu[0][2]
            h = xywh_cpu[0][3]

            license_plate_crop = result.orig_img[int(y-h/2-5):int(y+h/2+5), int(x-w/2-5):int(x+w/2+5)]

            result = model([license_plate_crop])

            resultsDigits = modelDigits([license_plate_crop])  

            for id,r in enumerate(resultsDigits):
                boxes = r.boxes

                if(len(boxes.xywh)>0):
                    xywh_cpu = boxes.xywh.to('cpu').numpy()

                    print(xywh_cpu)
                    for coords in xywh_cpu:
                        x = coords[0]
                        y = coords[1]
                        w = coords[2]
                        h = coords[3]

                        #draw.rectangle([int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)], outline="red", width=2)
                        cropped_character = r.orig_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                        # print(read_license_plate(cropped_character))

                        resized_img = resize_with_padding(cropped_character)

                        input_arr = tf.keras.utils.img_to_array(resized_img)  # (height, width, channels)
                        img_tensor = np.expand_dims(input_arr, axis=0)

                        imageClassified = classification_model(img_tensor)
                        predictions = np.argmax(imageClassified, axis=1)

                        print("Prediction:", decoder[predictions[0]])
                        debug_imshow("Character",  np.array(resized_img), waitKey=True)  # display to screen

