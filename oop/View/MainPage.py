import tkinter as tk

from oop.View.Dialog import *

class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.img_size = 800
        self.columnconfigure(0, weight=1, minsize=int(img_size / 4))
		self.columnconfigure(2, weight=3, minsize=int((img_size + 50)/3))
		self.columnconfigure(3, weight=3, minsize=int((img_size + 50)/3))
		self.columnconfigure(4, weight=4, minsize=int((img_size + 50)/3))
		self.rowconfigure(0, weight=1, minsize=int(img_size + 100))
		self.rowconfigure(1, weight=1, minsize=50)

        # initialize widgets
		self.photo_img = None
		self.img_size = img_size
		self.listbox = self.init_listbox()
		self.listbox.bind('<Double-1>', self.contolller.show_selection)
		self.img_panel = self.init_img_panel()
		self.init_radiobuttons()
		# init recalibrate button
		# self.recalibrate = 0
		Button(self, text="Recalibrate", command=self.controller.recalibrate).grid(row=1, column=4, sticky='nsew')
		self.update()

		# initialize target via a dialog window
		d_option = DialogOption(self, title="Select target")
		self.wait_window(d_option.top)

