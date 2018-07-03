import os
import sys
import glob
from tqdm import tqdm
from PIL import Image

def clean_images(dataset_dir, sizes, output_dir):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	image_list = glob.glob(os.path.join(dataset_dir, '*.jpg'))
	n_processed = 0
	n_rejected = 0	

	for idx, image in enumerate(tqdm(iterable=image_list, ncols=100)):
		
		try:
			im = Image.open(image).convert('RGB')
			w, h = im.size	

			if w == h:
				for size in sizes:
					im_thumb = im.copy()
					im_thumb.thumbnail(size, Image.ANTIALIAS)
					im_thumb.save(os.path.join(output_dir, "{0}{1}_{2}.jpg".format(dataset_dir, n_processed, size[0])), 'JPEG')
					n_processed += 1
			else:
				n_rejected += 1

		except Exception as e:
			n_rejected += 1
		
	print("* Processed {0} album covers | Rejected {1} album cover.".format(n_processed, n_rejected))

img_dirs = ['a', 'b', 'c', 's']
sizes = [(28, 28), (128, 128)]

for img_dir in img_dirs:
	clean_images(img_dir, sizes, 'covers')