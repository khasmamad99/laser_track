from oop.Controller.Controller import Controller
from oop.Controller.ViewController import ViewController
from oop.Model.User import User


if __name__ == "__main__":
    user = User()
    contoller = Controller(user)
    contoller.view_controller.mainloop()