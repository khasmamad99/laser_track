import cv2
import numpy as np
import time
import math
import imutils
from PIL import Image
from multiprocessing.pool import ThreadPool
import itertools as it


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6



def letterbox_image(image, size):
	'''resize image with unchanged aspect ratio using padding'''
	iw, ih = image.size
	w, h = size
	scale = min(w/iw, h/ih)
	nw = int(iw*scale)
	nh = int(ih*scale)

	image = image.resize((nw, nh), Image.BICUBIC)
	new_image = Image.new('RGB', size, (0, 0, 0))
	new_image.paste(image, ((w-nw)//2, (h-nh)//2))
	return new_image



def draw_score(frame, score, dist=None):
	text = "SCORE: " + str(score)
	if dist is not None:
		text = text + "    DISTANCE: " + format(dist, '.2f')
	org_y = frame.shape[0] - 10
	org_x = 10
	cv2.putText(frame, text, (int(org_x), int(org_y)),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def calc_score(conts, x, y):
	for i, cont in enumerate(conts):
		test = cv2.pointPolygonTest(cont, (x, y), True)
		if test >= 0:
			return 10 - i
	return 0


def calc_distance(pts):
	dist = 0
	start = False
	start_time = None
	prev_coords = None
	for pt in reversed(pts):
		coords, is_circle, _, time = pt
		if is_circle and not start:
			start = True
			prev_coords = coords
			start_time = time
		if start and (time - start_time < 3) and prev_coords is not None and coords is not None:
			dist += math.sqrt(math.pow((coords[0] - prev_coords[0]),
									   2) + math.pow((coords[1] - prev_coords[1]), 2))
		prev_coords = coords

	return dist


def detect_laser(image, dilate=False, erode=False, subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()):
	# resize image
	h, w, c = image.shape
	scale = min(640/w, 480/h)
	im = cv2.resize(image, (int(w*scale), int(h*scale)))

	# find mask
	mask = subtractor.apply(image, None, learningRate=0)
	kernel = np.ones((5, 5), np.uint8)
	if erode:
		mask = cv2.erode(mask, kernel, iterations=1)
	if dilate:
		mask = cv2.dilate(mask, kernel, iterations=1)

	cv2.imshow("mask", mask)

	# find avg
	avg = cv2.mean(mask)[0]
	if avg > 1 or avg < 0.0001:
		# if change in the image is too much (change in light/brightness), ignore it
		if avg > 1:
			print("Make sure that the camera is stable!")
		return False, None, None

	else:
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cnt = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
			cont = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [cnt], 0, (0, 255, 0), 3)
			cv2.imshow("cont", cont)
			print("AREA:", cv2.contourArea(cnt))
			if cv2.contourArea(cnt) > 5:
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				return True, cx, cy

			else:
				return False, None, None
		else:
			return False, None, None


def get_warped_coords(x, y, image, ref_img, h):
	# create a zero matrix with the same shape as the original image
	m = np.zeros(image.shape)
	# set the values of the elements in xth row and yth column to 255 in all image channels
	m[y, x, :] = 255
	# get the transformed image matrix
	m = cv2.warpPerspective(m, h, (ref_img.shape[1], ref_img.shape[0]))
	# coordinates of the nonzero elements of m correspond to the coordinates of x, y
	# in the warped (transformed) image
	nonzero = np.nonzero(m)
	if nonzero[0].size == 0 or nonzero[1].size == 0:
		return None
	else:
		y, x = [i[0] for i in nonzero[:-1]]
		return x, y


def init_feature(name):
	chunks = name.split('-')
	if chunks[0] == 'sift':
		detector = cv2.xfeatures2d.SIFT_create()
		norm = cv2.NORM_L2
	elif chunks[0] == 'surf':
		detector = cv2.xfeatures2d.SURF_create(800)
		norm = cv2.NORM_L2
	elif chunks[0] == 'orb':
		detector = cv2.ORB_create(400)
		norm = cv2.NORM_HAMMING
	elif chunks[0] == 'akaze':
		detector = cv2.AKAZE_create()
		norm = cv2.NORM_HAMMING
	elif chunks[0] == 'brisk':
		detector = cv2.BRISK_create()
		norm = cv2.NORM_HAMMING
	else:
		return None, None
	if 'flann' in chunks:
		if norm == cv2.NORM_L2:
			flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		else:
			flann_params = dict(algorithm=FLANN_INDEX_LSH,
								table_number=6,  # 12
								key_size=12,     # 20
								multi_probe_level=1)  # 2
		# bug : need to pass empty dict (#1329)
		matcher = cv2.FlannBasedMatcher(flann_params, {})
	else:
		matcher = cv2.BFMatcher(norm)
	return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append(kp1[m.queryIdx])
			mkp2.append(kp2[m.trainIdx])
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	return p1, p2, list(kp_pairs)


def affine_skew(tilt, phi, img, mask=None):
	'''
	affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
	Ai - is an affine transform matrix from skew_img to img
	'''
	h, w = img.shape[:2]
	if mask is None:
		mask = np.zeros((h, w), np.uint8)
		mask[:] = 255
	A = np.float32([[1, 0, 0], [0, 1, 0]])
	if phi != 0.0:
		phi = np.deg2rad(phi)
		s, c = np.sin(phi), np.cos(phi)
		A = np.float32([[c, -s], [s, c]])
		corners = [[0, 0], [w, 0], [w, h], [0, h]]
		tcorners = np.int32(np.dot(corners, A.T))
		x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
		A = np.hstack([A, [[-x], [-y]]])
		img = cv2.warpAffine(
			img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	if tilt != 1.0:
		s = 0.8*np.sqrt(tilt*tilt-1)
		img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
		img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0,
						 interpolation=cv2.INTER_NEAREST)
		A[0] /= tilt
	if phi != 0.0 or tilt != 1.0:
		h, w = img.shape[:2]
		mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
	Ai = cv2.invertAffineTransform(A)
	return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
	'''
	affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
	Apply a set of affine transformations to the image, detect keypoints and
	reproject them into initial image coordinates.
	See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.
	ThreadPool object may be passed to speedup the computation.
	'''
	params = [(1.0, 0.0)]
	for t in 2**(0.5*np.arange(1, 6)):
		for phi in np.arange(0, 180, 72.0 / t):
			params.append((t, phi))

	def f(p):
		t, phi = p
		timg, tmask, Ai = affine_skew(t, phi, img)
		keypoints, descrs = detector.detectAndCompute(timg, tmask)
		for kp in keypoints:
			x, y = kp.pt
			kp.pt = tuple(np.dot(Ai, (x, y, 1)))
		if descrs is None:
			descrs = []
		return keypoints, descrs

	keypoints, descrs = [], []
	if pool is None:
		ires = it.imap(f, params)
	else:
		ires = pool.imap(f, params)

	for i, (k, d) in enumerate(ires):
		print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
		keypoints.extend(k)
		descrs.extend(d)

	print()
	return keypoints, np.array(descrs)


def asift(img1, img2):
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	detector, matcher = init_feature('brisk-flann')

	pool = ThreadPool(processes=cv2.getNumberOfCPUs())
	kp1, desc1 = affine_detect(detector, img1, pool=pool)
	kp2, desc2 = affine_detect(detector, img2, pool=pool)

	raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

	if len(p1) >= 4:
		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
		print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
		# do not draw outliers (there will be a lot of them)
		kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
	else:
		H, status = None, None
		print('%d matches found, not enough for homography estimation' % len(p1))

	# Use homography
	height, width = img2.shape
	im1Reg = cv2.warpPerspective(img1, H, (width, height))

	return im1Reg, H