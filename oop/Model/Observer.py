from abc import ABC


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects (observables).
    """

    @abstractmethod
    def update(self, subject: Subject) -> None:
        """
        Receive update from subject.
        """
        pass