import argparse

import pytest

from steering_llms.cli import build_parser, parse_spec


def test_parse_spec():
    assert parse_spec("14:8949:7.5") == (14, 8949, 7.5)
    assert parse_spec("14:8949:-8") == (14, 8949, -8.0)


@pytest.mark.parametrize("bad", ["14:8949", "a:b:c", "1:2:3:4", ""])
def test_parse_spec_rejects(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_spec(bad)


def test_cli_args():
    a = build_parser().parse_args(["-s", "14:8949:-8", "--steer", "17:7426:5", "--quantize", "hi"])
    assert a.steer == [(14, 8949, -8.0), (17, 7426, 5.0)]
    assert a.quantize is True and a.prompt == "hi"
    assert build_parser().parse_args([]).quantize is None
    assert build_parser().parse_args(["--no-quantize"]).quantize is False
