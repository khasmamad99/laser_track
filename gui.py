from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk

main = Tk()
main.title('Laser Track')

scroll = Tk()

scroll.grid()

scrollbar = Scrollbar(scroll)
# scrollbar.grid(row = 0, column = 0, sticky = W, padx = 5, pady = 5, expand = YES)
scrollbar.pack( side = RIGHT, fill = Y )
mylist = Listbox(scroll, yscrollcommand = scrollbar.set ) 
for line in range(100): 
   mylist.insert(END, 'This is line number ' + str(line)) 
mylist.pack( side = LEFT, fill = BOTH, expand = YES )

scrollbar.config( command = mylist.yview )

main.mainloop()