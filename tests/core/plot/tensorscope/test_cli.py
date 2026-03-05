"""Tests for TensorScope CLI."""

from __future__ import annotations


def test_build_parser_supports_module_and_modules_subcommand():
    from cogpy.core.plot.tensorscope.cli import _build_parser

    ap = _build_parser()

    ns = ap.parse_args(["serve", "recording.nc", "--module", "psd_explorer"])
    assert ns.cmd == "serve"
    assert ns.module == "psd_explorer"

    ns2 = ap.parse_args(["modules"])
    assert ns2.cmd == "modules"


def test_cmd_modules_prints_known_modules(capsys):
    import pytest

    pytest.importorskip("numpy")

    from cogpy.core.plot.tensorscope.cli import _cmd_modules

    rc = _cmd_modules()
    assert rc == 0
    out = capsys.readouterr().out
    assert "psd_explorer" in out
