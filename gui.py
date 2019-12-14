from tkinter import *
from tkinter.ttk import *

from PIL import Image, ImageTk
from matplotlib import cm
import cv2

from multiprocessing import Process, Queue
from datetime import datetime

from laser_track import *


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


class GUI(Tk):
	def __init__(self, img_size=800, ref_img="target/target_reference_crop.jpg"):
		Tk.__init__(self)
		self.columnconfigure(0, weight=1, minsize=int(img_size / 4))
		self.columnconfigure(2, weight=3, minsize=int(img_size + 50))
		self.rowconfigure(0, weight=1, minsize=int(img_size + 100))
		self.resizable(0, 0)

		self.img_size = img_size
		self.listbox = self.init_listbox()
		self.img_panel = self.init_img_panel(ref_img)
		self.ref_img = ref_img

	def init_listbox(self):
		pady = 50
		scrollbar = Scrollbar(self)
		scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=pady)
		listbox = Listbox(self, yscrollcommand=scrollbar.set, font=('Arial', 12))
		listbox.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=pady)
		scrollbar.config(command=listbox.yview)
		return listbox

	def init_img_panel(self, img):
		img = cv2.imread(img, 0)
		ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
		img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)
		img = Image.fromarray(img)
		img = letterbox_image(img, (self.img_size, self.img_size))
		img = ImageTk.PhotoImage(img)
		panel = Label(self, image=img, borderwidth=20)
		panel.image = img
		panel.grid(row=0, column=2, sticky='nsew')
		return panel

	def update_image(self, img):
		self.img = ImageTk.PhotoImage(letterbox_image(
			 img, (self.img_size, self.img_size)))
		self.img_panel.configure(image=self.img)
		self.img_panel.image = self.img

	def insert_entry(self, entry):
		self.listbox.insert(END, entry)


def draw(ref_img, image, q):
	# align images
	ref_img = cv2.imread(ref_img)
	warped, h = align_images(image, ref_img)

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
	warped = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
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
		nonlocal drawing, x_prev, y_prev, start_time, pts, warped_copy, warped
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
				#gui.update_image(Image.fromarray(cv2.cvtColor(warped_copy, cv2.COLOR_BGR2RGB)))
				# set previous point to current point
				x_prev, y_prev = xi, yi
				# add current point and its relative time to the list
				pts.append(([x_prev, y_prev], time.time() - start_time))
				q.put((warped_copy.copy(), None))


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
			q.put((warped_copy.copy(), pts))
			print("points per sec:", count / 0.5)
			# reset pts
			pts = []

	# display original image and the warped image
	cv2.namedWindow("orig")
	#cv2.namedWindow("warped")
	cv2.setMouseCallback("orig", draw_line)
	cv2.imshow("orig", image)

	while(True):
		#cv2.imshow("warped", warped_copy)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()



root = GUI()
img = cv2.imread("target/2.jpeg")
q = Queue()
p = Process(target=draw, args=(root.ref_img, img, q))
p.start()

shoots = {}

def update_image():
	if not q.empty():
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		if pts is not None:
			now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
			shoots[now] = pts
			root.insert_entry(now)

	selection = root.listbox.curselection()
	root.after(1, update_image)

update_image()
root.mainloop()
p.join()








# img_size = 800

# root = Tk()
# root.columnconfigure(0, weight=1, minsize=int(img_size / 4))
# root.columnconfigure(2, weight=3, minsize=int(img_size + 50))
# root.rowconfigure(0, weight=1, minsize=int(img_size + 100))
# root.resizable(0,0)

# pady = 50
# scrollbar = Scrollbar(root)
# scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,10), pady=pady)
# listbox = Listbox(root, yscrollcommand=scrollbar.set, font=('Arial',12))
# for line in range(100): 
#    listbox.insert(END, 'This is line number ' + str(line))

# listbox.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=pady)
# scrollbar.config(command=listbox.yview)


# img = cv2.imread("aligned.jpg")
# # transform the image
# warped, transform_M, transform_size = transform_image(img)
# img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
# img = letterbox_image(img, (img_size, img_size))
# img = ImageTk.PhotoImage(img)
# panel = Label(root, image=img, borderwidth=20)
# panel.grid(row=0, column=2, sticky='nsew')

# root.mainloop()






# root.geometry("1200x800")
# root.resizable(0,0)

# pane_list = Frame(root, borderwidth=0, width=500, height=800)
# pane_list.pack(side=LEFT, fill=None, expand=False)
# pane_pic = Frame(root, borderwidth=0, width=700, height=800)
# pane_pic.pack(side=RIGHT, fill=None, expand=False)


# scrollbar = Scrollbar(pane_list)
# scrollbar.pack(expand=YES, side=RIGHT, fill=Y)
# listbox = Listbox(pane_list, yscrollcommand=scrollbar.set, font=('Comic Sans MS',10))
# for line in range(100): 
#    listbox.insert(END, 'This is line number' + str(line))

# listbox.pack(side=LEFT, fill=BOTH, expand=YES)
# scrollbar.config(command=listbox.yview)


# img = ImageTk.PhotoImage(Image.open("target.jpg"))
# panel = Label(pane_pic, image=img)
# panel.pack(expand=YES, anchor=CENTER, fill=BOTH)

# pane_list = Frame(root, borderwidth=5)
# pane_list.grid(column=0, row=0, columnspan=2)

# pane_pic = Frame(root, borderwidth=5)
# pane_pic.grid(row = 0, column = 2, sticky='nsew')
