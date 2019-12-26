import cv2
import numpy as np
import time
import imutils
from PIL import Image
from multiprocessing.pool import ThreadPool
import itertools as it


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def detect_laser(image, subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()):
	# resize image
	h, w, c = image.shape
	scale = min(640/w, 480/h)
	im = cv2.resize(image, (int(w*scale), int(h*scale)))

	# find mask
	mask = subtractor.apply(image)
	
	# find avg
	avg = cv2.mean(mask)[0]
	if avg > 1 or avg < 0.0001:
		# if change in the image is too much (change in light/brightness), ignore it
		return False, None, None

	else:
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cnt = sorted(contours, key = lambda x: cv2.contourArea(x), reverse=True)[0]
			if cv2.contourArea(cnt) > 10:
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				return  True, cx, cy

			else:
				return False, None, None
		else:
			return False, None, None

								

		
	


def subtract_frames(frame1, frame2):
	kernel = np.ones((5,5), np.uint8)
	gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	blur1 = cv2.GaussianBlur(gray1, (5,5), 0)
	dil1 = cv2.dilate(blur1, kernel, iterations=1) 
	gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	blur2 = cv2.GaussianBlur(gray2, (5,5), 0)
	dil2 = cv2.dilate(blur2, kernel, iterations=1) 
	subtracted = cv2.subtract(dil1, dil2)
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(subtracted)
	print(maxVal)
	if maxVal > 100:
		return True, maxLoc[0], maxLoc[1]
	else: return False, maxLoc[0], maxLoc[1]
	# nonzero = np.nonzero(sub)
	# if nonzero[0].size != 0 and nonzero[1].size != 0:
	# 	y, x = [i[0] for i in nonzero[:-1]]
	# 	return True, x, y
	# else: return False, 0, 0


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
			flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		else:
			flann_params= dict(algorithm = FLANN_INDEX_LSH,
							   table_number = 6, # 12
							   key_size = 12,     # 20
							   multi_probe_level = 1) #2
		matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
	else:
		matcher = cv2.BFMatcher(norm)
	return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append( kp1[m.queryIdx] )
			mkp2.append( kp2[m.trainIdx] )
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
		A = np.float32([[c,-s], [ s, c]])
		corners = [[0, 0], [w, 0], [w, h], [0, h]]
		tcorners = np.int32( np.dot(corners, A.T) )
		x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
		A = np.hstack([A, [[-x], [-y]]])
		img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	if tilt != 1.0:
		s = 0.8*np.sqrt(tilt*tilt-1)
		img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
		img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
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
	for t in 2**(0.5*np.arange(1,6)):
		for phi in np.arange(0, 180, 72.0 / t):
			params.append((t, phi))

	def f(p):
		t, phi = p
		timg, tmask, Ai = affine_skew(t, phi, img)
		keypoints, descrs = detector.detectAndCompute(timg, tmask)
		for kp in keypoints:
			x, y = kp.pt
			kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
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

	raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
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


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect		# top-left, top-right, bottom-right, bottom-left
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	transform_size = (maxWidth, maxHeight)
	warped = cv2.warpPerspective(image, M, transform_size)
	
	# return the warped image and transform matrix
	return warped, M, transform_size

def transform_image(image):
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	
	# convert the image to grayscale, blur it, and apply thresholding
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	# loop over the contours
	for i, c in enumerate(cnts):
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05 * peri, True)
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# apply the four point transform to obtain a top-down
	return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


def align_images(im1, im2):
 
	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	print(len(matches))

	# Draw top matches
	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
	cv2.imwrite("matches.jpg", imMatches)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))

	return im1Reg, h



def draw_laser(image):
	# # transform the image
	# warped, transform_M, transform_size = transform_image(image)

	# align images
	ref_img = cv2.imread("target/target_reference_crop.jpg")
	warped, h = align_images(image, ref_img)

	# copy warped image to preserve the original
	warped_copy = warped.copy()

	drawing = False		# control varibale for drawing
	x_prev, y_prev = -1, -1
	start_time = -1	
	pts = []			# list of tuples, where each tuple is ([x, y], time)

	def get_warped_coords(x, y):
		# create a zero matrix with the same shape as the original image
		m = np.zeros(image.shape)
		# set the values of the elements in xth row and yth column to 255 in all image channels
		m[y, x, :] = 255
		# get the transformed image matrix
		m = cv2.warpPerspective(m, h, (ref_img.shape[1], ref_img.shape[0]))
		# coordinates of the nonzero elements of m correspond to the coordinates of x, y
		# in the warped (transformed) image
		y, x = [i[0] for i in np.nonzero(m)[:-1]]
		return x, y

	# an event listener to draw the laser path
	def draw_line(event, x, y, flags, param):
		nonlocal drawing, x_prev, y_prev, start_time, pts, warped_copy,warped
		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			warped_copy = warped.copy()
			# set starting_time
			start_time = time.time()
			# get the first point	
			x_prev, y_prev = get_warped_coords(x, y)
			# add the starting point and its relative time to the list
			pts.append(([x_prev, y_prev], time.time() - start_time))	
		
		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				xi, yi = get_warped_coords(x, y)
				# draw a line between current and previous points
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,255,0), 1)		
				# set previous point to current point
				x_prev, y_prev = xi, yi
				# add current point and its relative time to the list
				pts.append(([x_prev, y_prev], time.time() - start_time))


		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			# redraw last 0.5s in red
			end_time = pts[-1][1]
			x_prev, y_prev = pts[-1][0]
			count = 1
			for p in reversed(pts[:-1]):
				if end_time - p[1] >= 0.5:
					break
				
				xi, yi = p[0]
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,0,255), 1)
				x_prev, y_prev = xi, yi
				count += 1
			print("points per sec:", count / 0.5)
			# reset pts
			pts = []

	# display original image and the warped image
	cv2.namedWindow("orig")
	cv2.namedWindow("warped")
	cv2.setMouseCallback("orig", draw_line)
	cv2.imshow("orig", image)

	while(True):
		cv2.imshow("warped", warped_copy)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()


# img = cv2.imread("target/2.jpeg")
# draw_laser(img)
