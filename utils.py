import cv2
import numpy as np
import time
import imutils
from PIL import Image


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


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
