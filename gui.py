from tkinter import *
from tkinter.ttk import *

from PIL import Image, ImageTk

root = Tk()
root.geometry("1200x800")
root.resizable(0,0)

pane_list = Frame(root, borderwidth=20, width=300, height=200)
pane_list.pack(side=LEFT, fill=BOTH, expand=YES)
pane_pic = Frame(root, borderwidth=20, width=800, height=800)
pane_pic.pack(side=RIGHT, fill=BOTH, expand=YES)


scrollbar = Scrollbar(pane_list)
scrollbar.pack(expand=False, side=RIGHT, fill=Y)
listbox = Listbox(pane_list, yscrollcommand=scrollbar.set, font=('Comic Sans MS',15))
for line in range(100): 
   listbox.insert(END, 'This is line number' + str(line))

listbox.pack(side=LEFT, fill=BOTH, expand=YES)
scrollbar.config(command=listbox.yview)


img = ImageTk.PhotoImage(Image.open("/home/khasmamad/Downloads/art-mickey-810x610.jpg"))
panel = Label(pane_pic, image=img)
panel.pack(expand=YES, anchor=CENTER, fill=BOTH)


root.mainloop()