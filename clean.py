import os
import sys
import glob
from PIL import Image

in_image_dir = 's'
out_image_dir = 's_out'
sm_size = 28, 28
lg_size = 128, 128
n_processed = 0
n_rejected = 0

if not os.path.isdir(out_image_dir):
	os.makedirs(out_image_dir)

for idx, image in enumerate(glob.glob(os.path.join(in_image_dir, '*.jpg'))):
	
	try:
		im = Image.open(image).convert('RGB')
		im_sm = im.copy()
		im_lg = im.copy()
		w, h = im.size	

		if w == h:
			im_sm.thumbnail(sm_size, Image.ANTIALIAS)
			im_lg.thumbnail(lg_size, Image.ANTIALIAS)
			n_processed += 1
			im_sm.save(os.path.join(out_image_dir, "sm_" + str(n_processed) + ".jpg"), 'JPEG')
			im_lg.save(os.path.join(out_image_dir, "lg_" + str(n_processed) + ".jpg"), 'JPEG')
			sys.stdout.write("* Processed {0: >8} album covers | Rejected {1: >8} album cover.\r".format(n_processed, n_rejected))
			sys.stdout.flush()

	except Exception as e:
		n_rejected += 1
	

