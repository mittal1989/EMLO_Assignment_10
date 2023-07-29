import requests
import json
import random
import time
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t","--text",type=int,required=True,help="Number of text quries to execute in GPT model")
parser.add_argument("-i","--images",type=int,required=True,help="Number of images to infer from VIT model")

args = parser.parse_args()

## -------------------- ##
## --- GPT Module ----- ##
## -------------------- ##

url = "http://15.207.109.7:8080/infer/"
# url = "http://127.0.0.1:8080/infer/"
def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
        if random.randrange(num):
            continue
        line = aline
    return line

exection_time = []
text_req = args.text
for i in range(text_req):

    f = open("./Data/Harry_Potter/Book1.txt",'rb')
    text = random_line(f)

    # print("\n Input {},  text --- {}".format(i, text))
    
    input = {"text": str(text),"max_new_tokens": 100}
    payload = json.dumps(input)
    headers = {
    'Content-Type': 'application/json'
    }
    t1 = time.time()
    response = requests.request("GET", url, headers=headers, data=payload)
    t2 = time.time()

    # print(response.text)
    # print("\n")

    exection_time.append((t2-t1) * 10**3)
  # print(response.text)

print("No of Text queries - {} \n Total Execution Time - {:.3f}ms \n Avg Execution Time   - {:.3f} ms \n".format(text_req,sum(exection_time),sum(exection_time)/len(exection_time)))


## -------------------- ##
## --- VIT Module ----- ##
## -------------------- ##
imgExtension = ["png", "jpeg", "jpg"] #Image Extensions to be chosen from
allImages = list()

def chooseRandomImage(directory="./Data/mnist_images"):
    for img in os.listdir(directory): #Lists all files
        ext = img.split(".")[len(img.split(".")) - 1]
        if (ext in imgExtension):
            allImages.append(img)
    choice = random.randint(0, len(allImages) - 1)
    chosenImage = allImages[choice] #Do Whatever you want with the image file
    return chosenImage



url = "http://15.207.109.7:8000/infer/"
# url = "http://127.0.0.1:8000/infer/"
exection_time = []

img_size = args.images
for i in range(img_size):
  # print("\n\n Image Counter {} \n".format(i))

  randomImage = chooseRandomImage()
  files=[
    ('image',('2.png',open("./Data/mnist_images/"+randomImage,'rb'),'image/png'))
  ]
  headers = {}
  t1 = time.time()
  response = requests.request("GET", url, headers=headers, files=files)
  t2 = time.time()
  exection_time.append((t2-t1) * 10**3)
  # print(response.text)


print("No of Images - {} \n Total Execution Time - {:.3f}ms \n Avg Execution Time   - {:.3f} ms \n".format(img_size,sum(exection_time),sum(exection_time)/len(exection_time)))