from ultralytics import YOLO
from imutils import paths
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

imagePaths = sorted(list(paths.list_images("./license_plates/group1")))
decoder = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

list_license_plates= []
cropped_licenses = []
model = YOLO("./license_detection.pt")
modelDigits = YOLO("./digit_detection_font4.pt")
modelClassifyDigits = YOLO("./digit_classification_X.pt")
classification_model = tf.saved_model.load("./best_model")

def debug_imshow(title, image, waitKey=False):
    cv2.imshow(title, image)

    if waitKey:
        cv2.waitKey(0)

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

                    resized_img = resize_with_padding(cropped_character)
                    input_arr = tf.keras.utils.img_to_array(resized_img)
                    img_tensor = np.expand_dims(input_arr, axis=0)

                    imageClassified = classification_model(img_tensor)
                    predictions = np.argmax(imageClassified, axis=1)

                    print("Prediction LENNET:", decoder[predictions[0]])
                    license_lennet += decoder[predictions[0]]

            print("Prediction LENNET:", license_lennet)
            print("Prediction YOLO:", license_yolo)

            debug_imshow("License Plate", license_plate_crop, waitKey=True)


