import inspect

import pytest
from typer.testing import CliRunner

from consts import MOVE_FACTOR
from motor_cli import app, experiment_edge_threshold, goto, monitor, resolve_displacement


def test_cli_defaults_to_hidden_experiment_move_factor() -> None:
    for command in (goto, monitor):
        option = inspect.signature(command).parameters["move_factor"].default
        assert option.default == MOVE_FACTOR
        assert option.hidden is True


def test_normalized_displacement_uses_experiment_workspace() -> None:
    edge_threshold = experiment_edge_threshold(640, 480, 30)

    assert edge_threshold == pytest.approx(135)
    assert resolve_displacement(-1, 1, 640, 480, edge_threshold) == pytest.approx(
        (-105, 105)
    )


def test_goto_uses_experiment_defaults_end_to_end() -> None:
    result = CliRunner().invoke(
        app, ["goto", "--gain", "1", "--dry-run", "--", "-0.5", "-0.5"]
    )

    assert result.exit_code == 0
    assert "ZM0P350M1P49M2P-483F" in result.stdout
    assert "--move-factor" not in CliRunner().invoke(app, ["goto", "--help"]).stdout
