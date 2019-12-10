from tkinter import *
from tkinter.ttk import *

from PIL import Image, ImageTk

root = Tk()
root.geometry("500x500")


pane_list = Frame(root, borderwidth=4)
pane_list.pack(expand=True, fill=Y, side=LEFT)
pane_pic = Frame(root, width=300, height=450)
pane_pic.pack(expand=True, fill=BOTH, side=RIGHT, padx=20, pady=20)


scrollbar = Scrollbar(pane_list, height=100)
scrollbar.pack(expand=False, side=RIGHT)
listbox = Listbox(pane_list, yscrollcommand=scrollbar.set, height=100)
for line in range(100): 
   listbox.insert(END, 'This is line number' + str(line))

listbox.pack(side=LEFT, fill=BOTH)
scrollbar.config(command=listbox.yview)

b1 = Button(pane_pic, text = "Click me !") 
b1.pack(expand = True, fill = BOTH, anchor=CENTER)

root.mainloop()