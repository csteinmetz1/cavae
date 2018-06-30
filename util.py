import os
import sys
import glob
#from PIL import Image

img_dir = 's_out'

for idx, image in enumerate(glob.glob(os.path.join(img_dir, 'lg_*.jpg'))):
	print(image)
	#im = Image.open(image)
	#sys.stdout.write("* Loaded {} album covers.\r".format(idx+1))
	#sys.stdout.flush()
