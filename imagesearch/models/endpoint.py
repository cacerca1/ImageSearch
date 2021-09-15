
import json
import requests

URL = 'https://or80tvm2cg.execute-api.us-east-1.amazonaws.com/test/search'

def infer(enc_img, k):
    """Infer using the AWS stack. 

    Args:
        enc_img (str): base64 encoded image
        k (int): number of desired similar images.

    Returns:
        list: list of similar images
    """

    params = {"k":k,"base64image": enc_img}

    resp = requests.post(URL, data=json.dumps(params))

    images = json.loads(resp.json()['body'])['images']
    # images = resp.json()['images'] # for get method with params key

    return [x.replace('&','&amp;') for x in images]# why is the replace necessary?
