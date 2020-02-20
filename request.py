import cv2
import os
import json
import cv2
import requests
import sys



def req2srv(imgs, endpoint):
    headers = {'content-type': 'application/json'}
    payload_temp = {
        "data": {
            "names": ["image"],
            "ndarray": []
        }
    }

    payload_temp["data"]['ndarray'] = imgs

    resp = requests.post(
        endpoint,
        data=json.dumps(payload_temp),
        headers=headers
    )

    if not resp.ok:
        print("ERROR EXIT FAILED")
        print(resp.content)
        quit(-100)
    result_dict = json.loads(resp.content.decode('utf-8'))
    rets = result_dict['data']['ndarray']

    return rets
    

def usage():
    print('usage : ' + sys.argv[0] + ' <img> <endpoint URL> ')
    print(' e.g {} {} {}'.format(sys.argv[0],'imgs/tdog17030770_TP_V.jpg', 'http://xxxxxxxxxxxxxxxx'))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        quit(-1)
    
    input_image = sys.argv[1]
    files = []
    files.append(input_image)

    endpoint = sys.argv[2]

    imgs = []
    image = cv2.imread(input_image)
    target_size = (224, 224)
    img_resized = cv2.resize(image, target_size)
    imgs.append(img_resized.tolist())

    rets = req2srv(imgs, endpoint)

    for img_file, ret in zip( files, rets):
        print("------ {} prediction results ------ ".format(img_file))
        for _, name, prob in ret:
            print("  {} : {}".format(name, prob))
        print("")
