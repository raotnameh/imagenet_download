from bs4 import BeautifulSoup
import numpy as np
import requests
import time
import PIL.Image
import urllib
import os
import csv
import numpy as np
import csv
import cv2
import argparse



parser = argparse.ArgumentParser(description = "1st draft of NIPS")
parser.add_argument("--id", default = "n02512053", type = str,  help = "enter the synset_id")
parser.add_argument("--dest", default = "/home/tnameh/linux/data/", type = str,  help = "enter destination")
parser.add_argument("--class-name", default = "fish", type = str,  help = "enter the classname")

args = parser.parse_args()

synset_name=args.id #enter the sysnet id of a specific class
destination_f=args.dest #enter the directory where to store the data
classname=args.class_name  #enter the classname
   
try:
	os.makedirs(destination_f+"/train")
	os.makedirs(destination_f+"/test")
	os.makedirs(destination_f+"/validation")

	print("Directory " , destination_f ,  " Created ")
except FileExistsError:
	print("Directory " , destination_f ,  " already exists")  


page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+synset_name)
soup = str(BeautifulSoup(page.content, 'html.parser'))

split_urls=soup.split('\r\n')#split 
print(" legth of the url found",len(split_urls))



#here I saved the links in csv file for total no of images of a particular class
with open(destination_f+"/"+classname+".csv", 'w',encoding="utf-8") as f:
    writer = csv.writer(f)
    for val in split_urls:
        writer.writerow([val])

#converting url to image 
def downloader(url):
    resp = urllib.request.urlopen(url)
    time.sleep(1); #remove this line if u want fast downloading trade-in for few images not being downloaded
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) #decoding colored image rgb
    return image

#opening csv file and storing urls in loadurl variable as list
with open(destination_f+"/"+classname+".csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    loadurl1 = list(reader)

#loading and filtering saved urls    
loadurl = list(filter(lambda x : x != [], loadurl1))


#train images(1000 expected)
_training_images=1000   #the number of training images to use
os.makedirs(destination_f+ "train/"+ classname+"/")
for progress in range(_training_images):
	if not loadurl[progress][0] == None:
		try:
			print(progress)
			I = downloader(loadurl[progress][0])
			if (len(I.shape))>1:
				save_path = destination_f+ "train/"+ classname+"/"+str(progress)+'.jpg'
				cv2.imwrite(save_path,I)
		except:
			None

#validdation(150 expected)
os.makedirs(destination_f+ "validation/"+ classname+"/")
for progress in range(150):
	if not loadurl[progress] == None:
		try:
			print(progress)
			I = downloader(loadurl[1000+progress][0])
			if (len(I.shape))>1:
				save_path = destination_f+ "validation/"+ classname+"/"+str(progress)+'.jpg'#validation folder has saved
				cv2.imwrite(save_path,I)
		except:
			None
	 
#test images(100 expected)
os.makedirs(destination_f+ "test/"+ classname+"/")
for progress in range(100):
	if not loadurl[progress] == None:
		try:
			print(progress)
			I = downloader(loadurl[1000+150+progress][0])
			if (len(I.shape))>1:
				save_path = destination_f+ "test/"+ classname+"/"+str(progress)+'.jpg'#test folder has saved
				cv2.imwrite(save_path,I)
		except:
			None