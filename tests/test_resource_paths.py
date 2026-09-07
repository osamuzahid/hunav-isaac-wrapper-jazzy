"""robot.yaml / USD / world overlay via SOCIAL_NAV_* (no Isaac Sim)."""

from __future__ import annotations

from hunav_isaac_wrapper import resource_paths as rp
from hunav_isaac_wrapper.robot_catalog import robots_config_root


def test_robots_config_root_honors_env(tmp_path, monkeypatch) -> None:
    (tmp_path / "stretch").mkdir()
    (tmp_path / "stretch" / "robot.yaml").write_text("name: stretch\n", encoding="utf-8")
    monkeypatch.setenv("SOCIAL_NAV_ROBOTS", str(tmp_path))
    assert robots_config_root() == tmp_path


def test_resolve_world_usd_prefers_overlay(tmp_path, monkeypatch) -> None:
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    usd = worlds / "museum.usd"
    usd.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("SOCIAL_NAV_WORLDS", str(worlds))
    got = rp.resolve_world_usd("museum", str(tmp_path / "missing"))
    assert got == str(usd)


def test_resolve_robot_file_usd_root(tmp_path, monkeypatch) -> None:
    usd_root = tmp_path / "robots"
    (usd_root / "reachy").mkdir(parents=True)
    target = usd_root / "reachy" / "reachy.usd"
    target.write_text("usd", encoding="utf-8")
    monkeypatch.setenv("SOCIAL_NAV_ROBOT_USD", str(usd_root))
    assert rp.resolve_robot_file("reachy/reachy.usd") == str(target)
