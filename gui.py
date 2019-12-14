from tkinter import *
from tkinter.ttk import *

from PIL import Image, ImageTk
from matplotlib import cm
import cv2

from laser_track import *


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

img_size = 800

root = Tk()
root.columnconfigure(0, weight=1, minsize=int(img_size / 4))
root.columnconfigure(2, weight=3, minsize=int(img_size + 50))
root.rowconfigure(0, weight=1, minsize=int(img_size + 100))
root.resizable(0,0)

pady = 50
scrollbar = Scrollbar(root)
scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,10), pady=pady)
listbox = Listbox(root, yscrollcommand=scrollbar.set, font=('Arial',12))
for line in range(100): 
   listbox.insert(END, 'This is line number ' + str(line))

listbox.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=pady)
scrollbar.config(command=listbox.yview)


img = cv2.imread("target/2.jpeg")
# transform the image
warped, transform_M, transform_size = transform_image(img)
img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
img = ImageTk.PhotoImage(letterbox_image(img, (img_size, img_size)))
panel = Label(root, image=img, borderwidth=20)
panel.grid(row=0, column=2, sticky='nsew')

root.mainloop()






#root.geometry("1200x800")
#root.resizable(0,0)

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