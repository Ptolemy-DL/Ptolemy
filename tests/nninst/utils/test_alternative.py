from nninst.utils.alternative import alt, alts

expected_configs = [
    ("windows", "clang", 4, 2),
    ("windows", "clang", 8, 4),
    ("windows", "msvc", 4, 2),
    ("windows", "msvc", 8, 4),
    ("linux", "gcc", 4, 2),
    ("linux", "gcc", 8, 4),
    ("linux", "clang", 4, 2),
    ("linux", "clang", 8, 4),
    ("macos", "gcc", 4, 2),
    ("macos", "gcc", 8, 4),
    ("macos", "clang", 4, 2),
    ("macos", "clang", 8, 4),
]


def test_alternative():
    os = alt("windows", "linux", "macos")
    compiler = alt("gcc", "clang", "msvc")
    ram = alt(4, 8, 16)
    cpu = alt(1, 2, 4)

    def configs():
        for config in (os * compiler).filter(
            lambda: not (os.value == "windows" and compiler.value == "gcc")
        ).filter(
            lambda: os.value == "windows" or compiler.value != "msvc"
        ):
            with config:
                for config in ram.only(4, 8) | cpu.only(2, 4):
                    with config:
                        yield os.value, compiler.value, ram.value, cpu.value

    assert list(configs()) == expected_configs


def test_subset():
    ram = alt(4, 8, 16)
    ram2 = ram.only(4, 8)
    assert ram.id == ram2.id
    assert list(ram.choices) == [4, 8, 16]
    assert list(ram2.choices) == [4, 8]


def test_zip():
    ram = alt(4, 8, 16)
    cpu = alt(1, 2, 4)

    def configs():
        for config in ram | cpu:
            with config:
                yield ram.value, cpu.value

    assert list(configs()) == [(4, 1), (8, 2), (16, 4)]


def test_product():
    os = alt("windows", "linux", "macos")
    compiler = alt("gcc", "clang", "msvc")
    ram = alt(4, 8)
    cpu = alt(2, 4)

    def configs():
        for config in (os * compiler * (ram | cpu)).filter(
            lambda: not (os.value == "windows" and compiler.value == "gcc")
        ).filter(
            lambda: os.value == "windows" or compiler.value != "msvc"
        ):
            with config:
                yield os.value, compiler.value, ram.value, cpu.value

    assert list(configs()) == expected_configs


def test_alts():
    os, compiler = alts(
        ["windows", "msvc"],
        ["linux", "gcc"],
        ["macos", "clang"],
    )
    assert list(os.choices) == ["windows", "linux", "macos"]
    assert list(compiler.choices) == ["msvc", "gcc", "clang"]
