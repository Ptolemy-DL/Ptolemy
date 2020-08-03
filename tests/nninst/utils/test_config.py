from nninst.utils import Config


def test_new():
    config = Config(a=1, b=2)
    assert config.a == 1
    assert config.b == 2
    assert config == {"a": 1, "b": 2}


def test_add():
    config1 = Config(a=1)
    config2 = Config(b=2)
    config = config1 + config2
    assert isinstance(config, Config)
    assert config == {"a": 1, "b": 2}
