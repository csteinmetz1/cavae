import os
import sys
import glob
import numpy as np
from scipy import misc

def load_data(split):
	img_dir = 's_out'
	covers = []

	for idx, image in enumerate(glob.glob(os.path.join(img_dir, 'sm_*.jpg'))):
		im = misc.imread(image)
		if im.shape != (28, 28, 3):
			print("Found invalid image shape.")
			print(im.shape, image)
		covers.append(misc.imread(image))
		sys.stdout.write("* Loaded {} album covers.\r".format(idx+1))
		sys.stdout.flush()

	split_idx = int(np.floor(len(covers) * split))

	return (np.asarray(covers[:split_idx]), np.asarray(covers[split_idx:]))
