class ViewProxy(Observable):

    _observers: List[Observer] = []
    _state: int = -1 
    current_image = None
    calibration: int = 0
    target_json: str = None


    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        try:
            self._observers.remove(observer)
        except ValueError, e:
            print("Could not detach the observer:", e)
        

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    def update_image(self, image) -> None:
        self._current_image = image
        self._state = 0
        self.notify()

    def set_calibration(self, val) -> None:
        assert val == 0 or val == 1, "unrecognized calibration value: " + str(val)
        self._calibration = val
        self._state = 1
        self.notify()
        