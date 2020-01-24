import tkinter as tk                
from tkinter import font  as tkfont
import datetime

from oop.View.MainPage import MainPage
# from oop.View.StartPage import StartPage
from oop.View.Dialog import *


class ViewController(tk.Tk):

    def __init__(self, controller, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.controller = controller
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (MainPage,):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainPage")


    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def show_selection(self, event):
        print("VIEW CONTROLLER SHOW SELECTION CALLED")
        self.frames["MainPage"].new_shot_button.config(state="normal")
        self.controller.show_selection()


    def init_target(self):
        self.target_dict = None
        self.wait_window(DialogOption(self, title="Select Target").top)
        return self.target_dict


    def update_attrs(self):
        shot, frame = self.controller.view_control.shot, self.controller.view_control.frame
        page = self.frames["MainPage"]
        if frame is not None:
            page.update_image(frame)
        if shot:
            self.controller.user.insert_shot(shot)
            page.insert_entry(datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y"))
            self.controller.view_control.shot = None

        self.after(1, self.update_attrs)


    def get_listbox_selection(self):
        return self.frames["MainPage"].listbox.curselection()

    
    def get_shot_type(self):
        return self.frames["MainPage"].rb_value.get()

    def get_new_shot(self):
        self.frames["MainPage"].new_shot_button.config(state="disabled")
        self.controller.get_new_shot()

    def enable_new_shot_button(self):
        self.frames["MainPage"].new_shot_button.config(state="disabled")

    def recalibrate(self):
        self.controller.recalibrate()