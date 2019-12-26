from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import cv2

from utils import letterbox_image

class Dialog:
	def __init__(self, parent, options=[""], title=None):
		self.parent = parent
		self.top = Toplevel(parent)
		self.top.columnconfigure(0, minsize=200)
		self.top.rowconfigure(0, minsize=50)
		self.top.rowconfigure(1, minsize=50)
		self.top.resizable(0,0)

		# create option menu
		self.option = StringVar(self.top)
		self.optmenu = OptionMenu(self.top, self.option, options[0], *options)
		self.optmenu.grid(row = 0, column = 0, sticky = 'nsew', padx=5, pady=5)

		# create the button
		b = Button(self.top, text="OK", command=self.ok)
		b.grid(row = 1, column = 0, sticky='ns', padx=5, pady=5)

	def ok(self):
		self.parent.target_img = "target/" + self.option.get() + ".jpg"
		self.top.destroy()



class GUI(Tk):
	def __init__(self, img_size=800):
		Tk.__init__(self)
		self.columnconfigure(0, weight=1, minsize=int(img_size / 4))
		self.columnconfigure(2, weight=3, minsize=int(img_size + 50))
		self.rowconfigure(0, weight=1, minsize=int(img_size + 100))
		self.resizable(0, 0)

		self.shoots = {}
		self.count = {}

		# initialize widgets
		self.photo_img = None
		self.img_size = img_size
		self.listbox = self.init_listbox()
		self.listbox.bind('<Double-1>', self.show_selection)
		self.img_panel = self.init_img_panel()
		self.update()


		# initialize target via a dialog window
		self.target_img = None
		options = ["circular1", "human1", "human2"]
		d = Dialog(self, options=options)
		self.wait_window(d.top)
		self.update_image(Image.open(self.target_img))


	def init_listbox(self):
		pady = 50
		scrollbar = Scrollbar(self)
		scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=pady)
		listbox = Listbox(self, yscrollcommand=scrollbar.set, font=('Arial', 12))
		listbox.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=pady)
		scrollbar.config(command=listbox.yview)
		return listbox

	def init_img_panel(self):
		self.photo_img = ImageTk.PhotoImage(letterbox_image(Image.fromarray(cv2.imread("white.jpg", 0)), (self.img_size, self.img_size)))
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

	def show_selection(self, event):
		selection = self.listbox.curselection()
		if selection:
			target_copy = self.target.copy()
			pts = self.shoots[selection[0]]
			prev = pts[0][0]
			red = False
			for i in range(1, len(pts)):
				pt = pts[i]
				if prev is None:
					prev = pt[0]
				if pt[0] is not None and prev is not None:
					if pt[1]:
						cv2.circle(target_copy, pt[0], 15, (255,0,0), -1)
						prev = pts[i+1][0]
						red = True
					else:
						if red: cv2.line(target_copy, prev, pt[0], (0,0,255), 2)
						else: cv2.line(target_copy, prev, pt[0], (0,255, 0), 2)
						prev = pt[0]

			self.update_image(Image.fromarray(cv2.cvtColor(target_copy, cv2.COLOR_BGR2RGB)))

# to do: add a button for new shot
# save the initial target
