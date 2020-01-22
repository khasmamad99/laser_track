from tkinter import Tk


class MainFrame:
    
    def __init__(self, master, controller):
        self.master = master
        self.controller = controller




    def update_frame(self, frame=None):
        if frame is None:
            frame = self.controller.veiw_attrs.frame
        
        # set new frame
        # self.after(1, self.update_frame)


