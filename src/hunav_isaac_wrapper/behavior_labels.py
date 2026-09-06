"""
Viewport overlays naming HuNav agent behavior types (operator demos).

Uses OmniGraph DrawLabel (omni.graph.visualization.nodes) anchored above each
agent. Viewport UI only — not visible to cameras/sensors.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# Behavior type → (label name, RGBA)
_BEH_STYLE: Dict[int, Tuple[str, Tuple[float, float, float, float]]] = {
    1: ("REGULAR", (1.0, 1.0, 1.0, 1.0)),
    2: ("IMPASSIVE", (0.75, 0.75, 0.75, 1.0)),
    3: ("SURPRISED", (1.0, 0.9, 0.2, 1.0)),
    4: ("SCARED", (1.0, 0.55, 0.15, 1.0)),
    5: ("CURIOUS", (0.2, 0.9, 1.0, 1.0)),
    6: ("THREATENING", (1.0, 0.25, 0.25, 1.0)),
}

GRAPH_PATH = "/World/HuNavBehaviorLabels"
_LABEL_Z_OFFSET = 1.8
_LABEL_SIZE = 16.0

_identity_matrix = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def behavior_labels_enabled() -> bool:
    """
    On by default for windowed GUI; off for headless.
    Override: HUNAV_BEHAVIOR_LABELS=0|1 (also true/false/yes/no/on/off).
    """
    env = os.environ.get("HUNAV_BEHAVIOR_LABELS", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    h = os.environ.get("HUNAV_ISAAC_HEADLESS", "").strip().lower()
    if h in ("1", "true", "yes", "on"):
        return False
    if h in ("0", "false", "no", "off"):
        return True
    try:
        from .sim_app_config import build_simulation_config

        return not bool(build_simulation_config().get("headless", False))
    except Exception:
        return True


def format_label(agent_id: int, beh_type: int, beh_state: int = 0) -> str:
    name, _ = _BEH_STYLE.get(int(beh_type), (f"TYPE{beh_type}", (1.0, 1.0, 1.0, 1.0)))
    text = f"A{int(agent_id)} · {name}"
    if int(beh_state) != 0:
        text += " · REACTING"
    return text


def color_for_behavior(beh_type: int) -> Tuple[float, float, float, float]:
    return _BEH_STYLE.get(int(beh_type), (1.0, 1.0, 1.0, 1.0))[1]


class BehaviorLabelOverlay:
    """Create / update DrawLabel nodes above HuNav agents."""

    def __init__(self) -> None:
        self._enabled = False
        self._agent_ids: List[int] = []
        self._failed = False
        self._fail_logged = False
        self._created = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._created and not self._failed

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled and self._created:
            self._blank_all()

    def ensure_labels(self, agent_ids: List[int]) -> bool:
        """Build ActionGraph with one DrawLabel per agent id (1-based)."""
        if not self._enabled or self._failed:
            return False
        ids = [int(i) for i in agent_ids]
        if self._created and ids == self._agent_ids:
            return True
        try:
            import omni.usd
            import omni.graph.core as og

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                self._fail("no USD stage")
                return False
            if stage.GetPrimAtPath(GRAPH_PATH).IsValid():
                stage.RemovePrim(GRAPH_PATH)

            keys = og.Controller.Keys
            create_nodes = [("on_playback_tick", "omni.graph.action.OnPlaybackTick")]
            connect = []
            set_values = []
            for aid in ids:
                node_name = f"draw_label_{aid}"
                create_nodes.append(
                    (node_name, "omni.graph.visualization.nodes.DrawLabel")
                )
                connect.append(
                    (
                        "on_playback_tick.outputs:tick",
                        f"{node_name}.inputs:execIn",
                    )
                )
                set_values.extend(
                    [
                        (f"{node_name}.inputs:text", f"A{aid}"),
                        (f"{node_name}.inputs:size", _LABEL_SIZE),
                        (f"{node_name}.inputs:color", [1.0, 1.0, 1.0, 1.0]),
                        (f"{node_name}.inputs:transform", list(_identity_matrix)),
                        (
                            f"{node_name}.inputs:offset",
                            [0.0, 0.0, float(_LABEL_Z_OFFSET)],
                        ),
                    ]
                )

            og.Controller.edit(
                {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: create_nodes,
                    keys.CONNECT: connect,
                    keys.SET_VALUES: set_values,
                },
            )
            self._agent_ids = ids
            self._created = True
            print(
                f"[behavior_labels] overlay ready for agents {ids} at {GRAPH_PATH}",
                flush=True,
            )
            return True
        except Exception as exc:
            self._fail(str(exc))
            return False

    def update_label(
        self,
        agent_id: int,
        x: float,
        y: float,
        z: float,
        beh_type: int,
        beh_state: int = 0,
    ) -> None:
        if not self.enabled:
            return
        aid = int(agent_id)
        if aid not in self._agent_ids:
            return
        try:
            import omni.graph.core as og

            node_path = f"{GRAPH_PATH}/draw_label_{aid}"
            text = format_label(aid, beh_type, beh_state)
            color = list(color_for_behavior(beh_type))
            # Identity transform; place via offset (world meters).
            og.Controller.set(
                og.Controller.attribute(f"{node_path}.inputs:text"), text
            )
            og.Controller.set(
                og.Controller.attribute(f"{node_path}.inputs:color"), color
            )
            og.Controller.set(
                og.Controller.attribute(f"{node_path}.inputs:offset"),
                [float(x), float(y), float(z) + _LABEL_Z_OFFSET],
            )
            og.Controller.set(
                og.Controller.attribute(f"{node_path}.inputs:transform"),
                list(_identity_matrix),
            )
        except Exception as exc:
            if not self._fail_logged:
                self._fail(f"update failed: {exc}")

    def destroy(self) -> None:
        if not self._created:
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(GRAPH_PATH).IsValid():
                stage.RemovePrim(GRAPH_PATH)
        except Exception:
            pass
        self._created = False
        self._agent_ids = []

    def _blank_all(self) -> None:
        if not self._created:
            return
        try:
            import omni.graph.core as og

            for aid in self._agent_ids:
                path = f"{GRAPH_PATH}/draw_label_{aid}.inputs:text"
                og.Controller.set(og.Controller.attribute(path), "")
        except Exception:
            pass

    def _fail(self, msg: str) -> None:
        self._failed = True
        if not self._fail_logged:
            print(f"[behavior_labels] disabled ({msg})", flush=True)
            self._fail_logged = True
