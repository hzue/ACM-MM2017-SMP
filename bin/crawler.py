from multiprocessing import Pool
import multiprocessing
from PIL import Image
from io import BytesIO
import requests
import re

def parse_task(flickr_url, index):
    res = requests.get(flickr_url)
    if res.status_code == 404: return
    img_url = re.findall('<meta property=\"og:image\" content=\"(.*)\" ', \
            res.content.decode('utf-8'))
    if len(img_url) == 0: return
    res = requests.get(img_url[0])
    if res.status_code == 404: return
    img = Image.open(BytesIO(res.content))
    img.save('image/{}.jpg'.format(index))
    return

def read_url_list(url_path):
    with open(url_path) as f:
        return f.read().splitlines()

def main():
    url_list = read_url_list('dataset/t1_train_image_link.txt')
    pool = Pool(100)
    for i in pool.starmap(parse_task, zip(url_list, range(len(url_list)))):
        print(i)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
