import cv2
import os
import json
import cv2
import requests
from VGGClassifier import VGGClassifier

payload_temp = {
    "data": {
        "names": ["image"],
        "ndarray": []
    }
}

if __name__ == '__main__':
    vgg = VGGClassifier()

    imgs = []
    files = os.listdir('imgs/') 
    for img_file in files:
        print(img_file)
        image = cv2.imread('imgs/' + img_file)
        imgs.append(image.tolist())

    payload_temp["data"]['ndarray'] = imgs
    rets = vgg.predict(payload_temp['data']['ndarray'], "piyopiyo")

    for img_file, ret in zip( files, rets):
        print("------ {} prediction results ------ ".format(img_file))
        for _, name, prob in ret:
            print("  {} : {}".format(name, prob))
        print("")
