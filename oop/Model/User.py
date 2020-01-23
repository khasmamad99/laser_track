# from oop.Model.Shot import Shot

class User(Observable):

    _shots: List[Shot] = []
    _observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)


    def detach(self, observer: Observer) -> None:
        try:
            self._observers.remove(observer)
        except ValueError:
            print("Could not detach the observer")
        

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)


    def insert_shot(self, shot: Shot) -> None:
        self._shots.append(shot)
        self.notify()