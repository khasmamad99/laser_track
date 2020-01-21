from multiprocessing import Process, Queue, Value, Manager

from tkinter.ttk import *
from PIL import Image, ImageTk

from tkinter import Tk, Label, Button

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.title = "nothing"

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")
        self.master.after(1000, self.greet)

    def bye(self):
        print("Bye!")
        self.master.after(1000, self.bye)

class GuiProxy:

    def __init__(self):
        self.gui_list = []
        root = Tk()
        self.title = "Old Title"
        self.gui_list.append(MyFirstGUI(root))


def update_title(self, ns):
    ns.gui.title = "New Title"

root = Tk()
gui = MyFirstGUI(root)
# root.after(2000, gui.greet)
# root.after(2000, gui.bye)
root.mainloop()

# pr = GuiProxy()
# print(pr.title)

# manager = Manager()
# ns =  manager.Namespace()
# ns.gui = pr
# p = Process(target=update_title, args=(ns))
# p.start()
# p.join()
# print("after:", pr.title)


