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



class Dialog:

	def __init__(self, parent, title=None):
		self.top = Toplevel(parent)
		self.top.columnconfigure(0, minsize=200)
		self.top.rowconfigure(0, minsize=50)
		self.top.rowconfigure(1, minsize=50)
		self.top.resizable(0,0)

		# create option menu
		self.init_opt = StringVar(self.top)
		self.init_opt.set("circular1")
		self.optmenu = OptionMenu(self.top, self.init_opt, "circular1", "circular1", "circular2", "circular3")
		self.optmenu.grid(row = 0, column = 0, sticky = 'nsew', padx=5, pady=5)

		# create the button
		b = Button(self.top, text="OK", command=self.ok)
		b.grid(row = 1, column = 0, sticky='ns', padx=5, pady=5)

	def ok(self):
		print("selected: ", self.init_opt.get)
		self.top.destroy()




class GUI(Tk):
	def __init__(self, img_size=800, ref_img="warped.jpg"):
		Tk.__init__(self)
		self.columnconfigure(0, weight=1, minsize=int(img_size / 4))
		self.columnconfigure(2, weight=3, minsize=int(img_size + 50))
		self.rowconfigure(0, weight=1, minsize=int(img_size + 100))
		self.resizable(0, 0)

		self.target = None
		self.photo_img = None
		self.ref_img = ref_img
		self.img_size = img_size
		self.listbox = self.init_listbox()
		self.img_panel = self.init_img_panel()
		self.update()
		d = Dialog(self)
		self.wait_window(d.top)


	def init_listbox(self):
		pady = 50
		scrollbar = Scrollbar(self)
		scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=pady)
		listbox = Listbox(self, yscrollcommand=scrollbar.set, font=('Arial', 12))
		listbox.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=pady)
		scrollbar.config(command=listbox.yview)
		return listbox

	def init_img_panel(self):
		# img = cv2.imread(img, 0)
		# img = cv2.GaussianBlur(img,(5,5),0)
		# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
		# img = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)
		# img = Image.fromarray(img)
		# img = letterbox_image(img, (self.img_size, self.img_size))
		self.target = cv2.imread("white.jpg", 0)
		self.photo_img = ImageTk.PhotoImage(letterbox_image(Image.fromarray(self.target), (self.img_size, self.img_size)))
		panel = Label(self, image=self.photo_img, borderwidth=20)
		panel.image = self.photo_img
		panel.grid(row=0, column=2, sticky='nsew')
		return panel

	def update_image(self, img):
		self.photo_img = ImageTk.PhotoImage(letterbox_image(
			 img, (self.img_size, self.img_size)))
		self.img_panel.configure(image=self.photo_img)
		self.img_panel.image = self.photo_img

	def insert_entry(self, entry):
		self.listbox.insert(END, entry)


def draw(ref_img, image, q):
	# align images
	ref_img = cv2.imread(ref_img)
	warped, h = align_images(image, ref_img)

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
	warped = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
	q.put(warped.copy())
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
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,255,0), 2)
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
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,0,255), 2)
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
root.target = q.get()
root.update_image(Image.fromarray(root.target))
shoots = {}
count = 0


def show_selection(event):
	selection = root.listbox.curselection()
	if selection:
		target_copy = root.target.copy()
		pts = shoots[selection[0]]
		end_time = pts[-1][1]
		x_prev, y_prev = pts[-1][0]
		count = 1
		for p in reversed(pts[:-1]):
			if end_time - p[1] >= 0.5:
				color = (0,255,0)
			else:
				color = (0,0,255)
			xi, yi = p[0]
			cv2.line(target_copy, (xi, yi), (x_prev, y_prev), color, 2)
			x_prev, y_prev = xi, yi
			count += 1

		root.update_image(Image.fromarray(cv2.cvtColor(target_copy, cv2.COLOR_BGR2RGB)))



def update_image():
	global count, shoots
	if not q.empty():
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		if pts is not None:
			now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
			shoots[count] = pts
			count += 1
			root.insert_entry(now)

	root.after(1, update_image)

root.listbox.bind('<Double-1>', show_selection)
update_image()
root.mainloop()
p.join()