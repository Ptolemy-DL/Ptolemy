import itertools
from typing import Generic, Iterable, Iterator, List, TypeVar, Union

from nninst.utils import Config, context, merge_dict

T = TypeVar("T")


class Alternative(Generic[T]):
    def __init__(self, *choices: T):
        self.choices = choices
        self._id = id(self)

    @property
    def id(self):
        return self._id

    def only(self, *choices: T) -> "Subset[T]":
        return Subset(self, choices)

    def __iter__(self) -> Iterator[Config]:
        for choice in self.choices:
            yield Config({self.id: choice})

    @property
    def value(self) -> T:
        for config in context.context().configs:
            if self.id in config:
                return config[self.id]
        raise RuntimeError(f"alternative{self.choices} not found in all configs")

    def __getitem__(self, config: Config) -> T:
        return config[self.id]

    def __mul__(self, other: Union["Alternative", "Zip", "Product"]) -> "Product":
        if isinstance(other, Alternative):
            return Product(alternatives=[self, other])
        elif isinstance(other, (Zip, Product)):
            return other * self
        else:
            raise TypeError()

    def __or__(self, other: Union["Alternative", "Zip"]) -> "Zip":
        if isinstance(other, Alternative):
            return Zip(alternatives=[self, other])
        elif isinstance(other, Zip):
            return other | self
        else:
            raise TypeError()


class Subset(Alternative[T]):
    def __init__(self, base: Union[Alternative[T], "Subset[T]"], choices: Iterable[T]):
        assert set(choices).issubset(base.choices)
        super().__init__(*choices)
        if isinstance(base, Alternative):
            self.base = base
        else:
            self.base = base.base

    @property
    def id(self):
        return self.base.id


def alt(*choices):
    return Alternative(*choices)


def alts(*zips):
    return tuple(alt(*choices) for choices in zip(*zips))


class Zip:
    def __init__(self, alternatives: List[Alternative] = None):
        self.alternatives = alternatives or []
        if len(self.alternatives) != 0:
            lens = list(map(lambda _: len(_.choices), self.alternatives))
            assert all(map(lambda x: x == lens[0], lens))

    def __iter__(self) -> Iterator[Config]:
        ids = list(map(lambda alt: alt.id, self.alternatives))
        for choices in zip(*list(map(lambda alt: alt.choices, self.alternatives))):
            yield Config(dict(zip(ids, choices)))

    def __or__(self, other: Union["Alternative", "Zip"]) -> "Zip":
        if isinstance(other, Alternative):
            return Zip(alternatives=[*self.alternatives, other])
        elif isinstance(other, Zip):
            return Zip(alternatives=[*self.alternatives, *other.alternatives])
        else:
            raise TypeError()

    def __mul__(self, other: Union["Alternative", "Zip", "Product"]) -> "Product":
        if isinstance(other, Alternative):
            return Product(alternatives=[other], zips=[self])
        elif isinstance(other, Zip):
            return Product(zips=[self, other])
        elif isinstance(other, Product):
            return other * self
        else:
            raise TypeError()


class Product:
    def __init__(
        self,
        alternatives: List[Alternative] = None,
        zips: List[Zip] = None,
        filters=None,
    ):
        self.alternatives = alternatives or []
        self.zips = zips or []
        self.filters = filters or []

    def __iter__(self) -> Iterator[Config]:
        configs_from_zips = list(itertools.product(*[list(zip) for zip in self.zips]))
        for choices in itertools.product(
            *list(map(lambda _: _.choices, self.alternatives))
        ):
            config = Config(dict(zip(map(lambda _: _.id, self.alternatives), choices)))
            for configs in configs_from_zips:
                merged_config = Config(merge_dict(config, *configs))
                with merged_config:
                    if all(filter_fn() for filter_fn in self.filters):
                        yield merged_config

    def __mul__(self, other: Union["Alternative", "Zip", "Product"]) -> "Product":
        if isinstance(other, Alternative):
            return Product(
                alternatives=[*self.alternatives, other],
                zips=self.zips,
                filters=self.filters,
            )
        elif isinstance(other, Zip):
            return Product(
                alternatives=self.alternatives,
                zips=[*self.zips, other],
                filters=self.filters,
            )
        elif isinstance(other, Product):
            return Product(
                alternatives=[*self.alternatives, *other.alternatives],
                zips=[*self.zips, *other.zips],
                filters=[*self.filters, *other.filters],
            )
        else:
            raise TypeError()

    def filter(self, filter_fn) -> "Product":
        return Product(
            alternatives=self.alternatives,
            zips=self.zips,
            filters=[*self.filters, filter_fn],
        )
