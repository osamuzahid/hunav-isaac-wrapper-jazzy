"""robot.yaml catalog discovery (no Isaac Sim, no GPU)."""

from __future__ import annotations

from pathlib import Path

import hunav_isaac_wrapper.robot_catalog as catalog


def test_preferred_names_omit_stretch_wheeled() -> None:
    src = Path(catalog.__file__).read_text(encoding="utf-8")
    assert "stretch_wheeled" not in src


def test_list_lab_robot_names_from_folders(tmp_path, monkeypatch) -> None:
    (tmp_path / "reachy").mkdir()
    (tmp_path / "reachy" / "robot.yaml").write_text("name: reachy\n", encoding="utf-8")
    (tmp_path / "stretch").mkdir()
    (tmp_path / "stretch" / "robot.yaml").write_text("name: stretch\n", encoding="utf-8")
    (tmp_path / "stretch_wheeled").mkdir()
    (tmp_path / "stretch_wheeled" / "robot.yaml").write_text(
        "name: stretch_wheeled\n", encoding="utf-8"
    )
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "robot.yaml").write_text("name: other\n", encoding="utf-8")
    (tmp_path / "empty_dir").mkdir()

    monkeypatch.setattr(catalog, "robots_config_root", lambda: tmp_path)
    names = catalog.list_lab_robot_names()
    assert names[0] == "stretch"
    assert names[1] == "reachy"
    assert "other" in names
    assert "stretch_wheeled" in names  # discovered if a folder exists, not preferred-first
    assert names.index("stretch_wheeled") > names.index("reachy")
    assert "empty_dir" not in names


def test_list_robot_choices_upstream_first(tmp_path, monkeypatch) -> None:
    (tmp_path / "stretch").mkdir()
    (tmp_path / "stretch" / "robot.yaml").write_text("name: stretch\nusd_package_file: x\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "robots_config_root", lambda: tmp_path)
    choices = catalog.list_robot_choices(["jetbot", "carter"])
    assert choices[:2] == ["jetbot", "carter"]
    assert "stretch" in choices
