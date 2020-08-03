from .context import context
from .utils import merge_dict


class Config(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __add__(self, other: "Config") -> "Config":
        return Config(merge_dict(self, other))

    def __enter__(self):
        context().configs.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        context().configs.pop()
