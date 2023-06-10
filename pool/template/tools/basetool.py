from abc import ABC, abstractmethod


class BaseTool(ABC):

    @abstractmethod
    def get_tool_name(self) -> str:
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def default_output(self) -> str:
        # default output when failed
        pass


