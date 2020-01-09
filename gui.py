from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import json
from utils import letterbox_image, draw_score


class Dialog:
	def __init__(self, parent, title=None):
		self.parent = parent
		self.top = Toplevel(parent)
		self.top.columnconfigure(0, minsize=200)
		self.top.rowconfigure(0, minsize=50)
		self.top.rowconfigure(1, minsize=50)
		self.top.resizable(0,0)
		if title:
			self.top.title(title)
		self.top.geometry("+%d+%d" % (parent.winfo_rootx()+int(parent.img_size/2),
                                  parent.winfo_rooty()+int(parent.img_size/2)))



class DialogOption(Dialog):
	def __init__(self, parent, title=None, options=[]):
		Dialog.__init__(self, parent, title)

		# create option menu
		self.option = StringVar(self.top)
		if not options:
			self.targets = json.load(open("target/targets.json", 'r'))
			options = list(self.targets.keys())
		self.optmenu = OptionMenu(self.top, self.option, options[0], *options)
		self.optmenu.grid(row = 0, column = 0, sticky = 'nsew', padx=5, pady=5)

		# create the button
		ok_b = Button(self.top, text="OK", command=self.ok)
		ok_b.grid(row = 1, column = 0, sticky='ns', padx=5, pady=5)

	def ok(self):
		self.parent.target_name = self.option.get()
		self.parent.target_dict = self.options[self.option.get()]
		self.top.destroy()



class DialogRecalib:
	def __init__(self, parent, title=None, text = ""):
		Dialog.__init__(self, parent, title)

		# create label
		Label(self.top, text = text).grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
		# create button
		ok_b = Button(self.top, text="OK", command=self.ok)
		ok_b.grid(row = 1, column = 0, sticky='ns', padx=5, pady=5)

	def ok(self):
		self.top.destroy()



class GUI(Tk):
	def __init__(self, img_size=800):
		Tk.__init__(self)
		self.columnconfigure(0, weight=1, minsize=int(img_size / 4))
		self.columnconfigure(2, weight=3, minsize=int((img_size + 50)/3))
		self.columnconfigure(3, weight=3, minsize=int((img_size + 50)/3))
		self.columnconfigure(4, weight=4, minsize=int((img_size + 50)/3))
		self.rowconfigure(0, weight=1, minsize=int(img_size + 100))
		self.rowconfigure(1, weight=1, minsize=50)
		self.resizable(0, 0)

		self.shots = {}
		self.count = 0

		# initialize widgets
		self.photo_img = None
		self.img_size = img_size
		self.listbox = self.init_listbox()
		self.listbox.bind('<Double-1>', self.show_selection)
		self.img_panel = self.init_img_panel()
		self.init_radiobuttons()
		# init recalibrate button
		self.recalibrate = 0
		Button(self, text="Recalibrate", command=self.recalib).grid(row=1, column=4, sticky='nsew')
		self.update()

		# initialize target via a dialog window
		self.target_name = None
		self.target_dict = None
		d_option = DialogOption(self, title="Select target")
		self.wait_window(d_option.top)
		self.update_image(Image.open(self.target["img_path"]))


	def recalib(self):
		self.recalibrate = 1
		d_recalib = DialogRecalib(self, title="Recalibrate", text="Shoot at the center to recalibrate")
		self.wait_window(d_recalib.top)


	def init_radiobuttons(self):
		self.rb_value = IntVar(self, 1)
		rb_one_shot = Radiobutton(self, text = "one-shot", variable = self.rb_value,  value=0)
		rb_one_shot.grid(row=1, column=2, sticky='nsew')
		rb_track = Radiobutton(self, text = "track", variable=self.rb_value, value=1)
		rb_track.grid(row=1, column=3, sticky='nsew')


	def init_listbox(self):
		pady = 50
		scrollbar = Scrollbar(self)
		scrollbar.grid(row=0, column=1, rowspan=2, sticky='ns', padx=(0, 10), pady=pady)
		listbox = Listbox(self, yscrollcommand=scrollbar.set, font=('Arial', 12))
		listbox.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(10, 0), pady=pady)
		scrollbar.config(command=listbox.yview)
		return listbox


	def init_img_panel(self):
		self.photo_img = ImageTk.PhotoImage(letterbox_image(Image.fromarray(cv2.imread("target/data/init_bg.jpg", 0)), (self.img_size, self.img_size)))
		panel = Label(self, image=self.photo_img, borderwidth=20)
		panel.image = self.photo_img
		panel.grid(row=0, column=2, columnspan=3, sticky='nsew')
		return panel


	def update_image(self, img):
		self.photo_img = ImageTk.PhotoImage(letterbox_image(
			 img, (self.img_size, self.img_size)))
		self.img_panel.configure(image=self.photo_img)
		self.img_panel.image = self.photo_img


	def insert_entry(self, entry, pts):
		self.shots[root.count] = pts
		self.count += 1
		self.listbox.insert(END, entry)


	def show_selection(self, event):
		selection = self.listbox.curselection()
		if selection:
			target = cv2.imread(self.target_img)
			pts = self.shots[selection[0]]
			prev = pts[0][0]
			red = False
			for i in range(0, len(pts)):
				pt = pts[i]
				if prev is None:
					prev = pt[0]
				if pt[0] is not None and prev is not None:
					if pt[1]:
						cv2.circle(target, pt[0], 15, (255,0,0), -1)
						scr = pt[2]
						dist = pt[3]
						draw_score(target, scr, dist)
						prev = None
						red = True
					else:
						if red: cv2.line(target, prev, pt[0], (0,0,255), 2)
						else: cv2.line(target, prev, pt[0], (0,255, 0), 2)
						prev = pt[0]

			self.update_image(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB)))

# to do: add a button for new shot
# save the initial target
