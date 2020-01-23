from tkinter import *
from tkinter.ttk import *

import json


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
		self.top.geometry("+%d+%d" % (parent.winfo_rootx(),
                                  parent.winfo_rooty()))


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
		self.parent.target_dict = self.targets[self.option.get()]
		self.top.destroy()



class DialogRecalib(Dialog):
	def __init__(self, parent, title=None, text = ""):
		Dialog.__init__(self, parent, title)

		# create label
		Label(self.top, text = text).grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
		# create button
		ok_b = Button(self.top, text="OK", command=self.ok)
		ok_b.grid(row = 1, column = 0, sticky='ns', padx=5, pady=5)

	def ok(self):
		self.top.destroy()