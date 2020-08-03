from nninst.utils import *


def test_filter_not_null():
    assert list(filter_not_null([1, 2, None])) == [1, 2]


def test_filter_value_not_null():
    assert filter_value_not_null({1: 2, 3: None}) == {1: 2}


def test_merge_dict():
    assert merge_dict({1: 2}, {3: 4}) == {1: 2, 3: 4}
