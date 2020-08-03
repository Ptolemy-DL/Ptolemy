from abc import ABC, abstractmethod

__all__ = ["CWAttack"]


class CWAttack(ABC):
    @abstractmethod
    def attack(self, imgs, targets):
        ...
