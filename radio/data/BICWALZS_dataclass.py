#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass, field, InitVar
import pandas as pd
import random
import math


@dataclass
class DataClass_Modern(object):

    # Invisible attribute (init-only)
    attr0: InitVar[int] = 81

    # Initialized attribute
    attr1: int = 0
    attr2: float = 0.0
    attr3: str = 'undefined'
    attr4: list = field(default_factory=list)

    # Generated attribute
    attr5: float = field(init=False)

    # Generated attribute - read property
    @property
    def attr5(self) -> float:
        return math.sqrt(abs(self._attrHidden))

    # Generated attr - set property (required by dataclasses)
    @attr5.setter
    def attr5(self, _):
        pass  # Do nothing, this is a read-only attribute

    def __post_init__(self, attr0):
        # Make a copy of the init-only attribute to a local attribute that
        # is used for the generated attribute (via a property)
        self._attrHidden = attr0  # This attribute should remain hidden from pandas

    @classmethod
    def rand_factory(cls):
        '''
        Returns an object of the calling class with randomized initialization attributess
        '''
        return cls(attr0=random.randint(-1e3, 1e3),
                   attr1=random.randint(-1e6, 1e6),
                   attr2=random.random(),
                   attr3=random.choice([
                       'Tool', 'PinkFloyd', 'Soundgarden', 'FaithNoMore',
                       'aPerfectCircle', 'KingCrimson', 'PearlJam',
                       'ChildrenOfBodom'
                   ]),
                   attr4=random.choices(range(100, 999), k=3))


if __name__ == '__main__':

    rand_objects = [DataClass_Modern.rand_factory() for _ in range(100)]
    df = pd.DataFrame(rand_objects)

    print(df)
