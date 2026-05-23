from projection import __version__, hello


def test_hello() -> None:
    assert hello() == "Hello from projection!"


def test_version() -> None:
    assert __version__ == "0.1.0"
