from abc import abstractmethod
from typing import Protocol, Any


class Transformer(Protocol):

    @abstractmethod
    def transform_1(self, data: str) -> Any:
        ...

    @abstractmethod
    def transform_2(self, data: str) -> Any:
        ...
