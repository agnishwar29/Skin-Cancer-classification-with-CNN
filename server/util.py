import cv2
import json
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import base64

__class_name_to_number = {}
__class_number_to_name = {}

__model = None



def load_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(r'D:\Python projects\Machine Learning\Skin Cancer Prediction\artifacts\class_dictionary.json','r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k, v in __class_name_to_number.items()}
    global __model

    model_path = r"D:\Python projects\Machine Learning\Skin Cancer Prediction\artifacts\skin_cancer.h5"
    __model = load_model(model_path)
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

#Test image
def get_bs4_test_image():
    with open("b64.txt") as f:
        return f.read()

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def get_data(image_path,image_base64_data):

    if image_path:
        img = image.load_img(image_path, target_size=(224,224))
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)


    return img


def classify_image(image_base64_data, file_path=None):
    img = get_data(file_path, image_base64_data)

    new_array = image.img_to_array(img)
    new_array = new_array.reshape(1, new_array.shape[0], new_array.shape[1], new_array.shape[2])
    final = preprocess_input(new_array)

    result = int(__model.predict(final)[0][0])
    result = class_number_to_name(result)
    return result

if __name__ == "__main__":
    load_artifacts()
    # print(classify_image(get_cv2_image_from_base64_string(), None))