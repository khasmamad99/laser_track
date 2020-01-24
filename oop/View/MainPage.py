import tkinter as tk
from tkinter.ttk import *
from PIL import Image as Img
from PIL import ImageTk
import cv2

from oop.View.Dialog import *
from oop.Controller.utils import letterbox_image

class MainPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		self.img_size = 800
		self.columnconfigure(0, weight=1, minsize=int(self.img_size / 4))
		self.columnconfigure(2, weight=1, minsize=int((self.img_size + 50)/4))
		self.columnconfigure(3, weight=1, minsize=int((self.img_size + 50)/4))
		self.columnconfigure(4, weight=1, minsize=int((self.img_size + 50)/4))
		self.columnconfigure(5, weight=1, minsize=int((self.img_size+50)/4))
		self.rowconfigure(0, weight=1, minsize=int(self.img_size + 100))
		self.rowconfigure(1, weight=1, minsize=50)

		# initialize widgets
		self.photo_img = None
		self.listbox = self.init_listbox()
		self.listbox.bind('<Double-1>', self.controller.show_selection)
		self.img_panel = self.init_img_panel()
		self.init_radiobuttons()
		self.new_shot_button = Button(self, text="New shot", state="disabled", command=self.controller.get_new_shot)
		self.new_shot_button.grid(row=1, column=5, sticky='nsew')
		Button(self, text="Recalibrate", command=self.controller.recalibrate).grid(row=1, column=4, sticky='nsew')
		self.update()

	
	def update_image(self, img):
		img = Img.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		self.photo_img = ImageTk.PhotoImage(letterbox_image(
			 img, (self.img_size, self.img_size)))
		self.img_panel.configure(image=self.photo_img)
		self.img_panel.image = self.photo_img


	def insert_entry(self, entry):
		self.listbox.insert(END, entry)


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
		self.photo_img = ImageTk.PhotoImage(letterbox_image(Img.fromarray(cv2.imread("target/data/init_bg.jpg", 0)), (self.img_size, self.img_size)))
		panel = Label(self, image=self.photo_img, borderwidth=20)
		panel.image = self.photo_img
		panel.grid(row=0, column=2, columnspan=4, sticky='nsew')
		return panel
