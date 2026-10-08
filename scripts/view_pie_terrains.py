# ruff: noqa: E402
"""Inspect every PIE Parkour terrain and difficulty in a CPU-only 3D gallery."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import trimesh
import tyro
import viser
from mjlab.viewer.viser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

from src.tasks.pie.parkour_terrains import PIE_PARKOUR_TERRAINS_CFG


@dataclass(frozen=True)
class PreviewConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    seed: int = 42
    shared_layout: bool = False
    """Reuse random draws across levels to compare difficulty on one layout."""
    spacing: float = 2.0
    """Display-only separation between tiles, in metres."""
    check: bool = False
    """Generate and validate all tiles without starting the viewer."""


@dataclass
class TerrainTile:
    name: str
    level: int
    mesh: trimesh.Trimesh
    offset: np.ndarray
    size: tuple[float, float]
    origin: np.ndarray
    goals: np.ndarray


def _terrain_mesh(model, geom_id):
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        return mujoco_mesh_to_trimesh(model, geom_id)
    mesh = create_primitive_mesh(model, geom_id)
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_HFIELD:
        return mesh
    # mjlab's converter draws only the top surface. Include the solid sides
    # so rough obstacle walls and gap boundaries remain visible in the gallery.
    field = model.geom_dataid[geom_id]
    rows, cols = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
    border = np.concatenate((
        np.arange(cols), np.arange(1, rows) * cols + cols - 1,
        (rows - 1) * cols + np.arange(cols - 2, -1, -1),
        np.arange(rows - 2, 0, -1) * cols,
    ))
    top = mesh.vertices[border]
    bottom = top.copy()
    bottom[:, 2] = -model.hfield_size[field, 3]
    count = len(border)
    index = np.arange(count)
    following = (index + 1) % count
    faces = np.vstack((np.column_stack((index, index + count, following)),
                       np.column_stack((following, index + count, following + count))))
    walls = trimesh.Trimesh(vertices=np.vstack((top, bottom)), faces=faces, process=False)
    walls.visual.vertex_colors = (model.geom_rgba[geom_id] * 255).astype(np.uint8)
    return trimesh.util.concatenate((mesh, walls))


def build_tiles(cfg: PreviewConfig) -> list[TerrainTile]:
    """Call the actual task generators at the exact playback levels 0/9..9/9."""
    if not np.isfinite(cfg.spacing) or cfg.spacing < 0:
        raise ValueError("--spacing must be finite and nonnegative.")
    if cfg.seed < 0:
        raise ValueError("--seed must be nonnegative.")
    generator = PIE_PARKOUR_TERRAINS_CFG
    length, width = generator.size
    tiles = []
    for column, (name, source) in enumerate(generator.sub_terrains.items()):
        for level in range(10):
            terrain = copy.deepcopy(source)
            terrain.size = generator.size
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            # Independent tiles expose placement/count variation by default.
            seed_parts = [cfg.seed, column] if cfg.shared_layout else [cfg.seed, column, level]
            rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
            output = terrain.function(level / 9, spec, rng)
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            meshes = []
            for geom_id in range(model.ngeom):
                mesh = _terrain_mesh(model, geom_id)
                transform = np.eye(4)
                transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
                transform[:3, 3] = data.geom_xpos[geom_id]
                mesh.apply_transform(transform)
                meshes.append(mesh)
            if not meshes:
                raise ValueError(f"Empty terrain: {name}, level {level}")
            mesh = trimesh.util.concatenate(meshes)
            origin = np.asarray(output.origin, dtype=float).copy()
            goals = np.asarray((output.flat_patches or {}).get("route_goals", []), dtype=float)
            if goals.ndim != 2 or goals.shape[1] != 3 or not len(goals):
                raise ValueError(f"Missing route goals: {name}, level {level}")
            if not all(np.isfinite(value).all() for value in (mesh.vertices, origin, goals)):
                raise ValueError(f"Non-finite terrain: {name}, level {level}")
            offset = np.array([
                (column - (len(generator.sub_terrains) - 1) / 2) * (length + cfg.spacing) - length / 2,
                (level - 4.5) * (width + cfg.spacing) - width / 2,
                0.0,
            ])
            tiles.append(TerrainTile(name, level, mesh, offset, generator.size, origin, goals))
    return tiles


def build_viewer(cfg: PreviewConfig, tiles: list[TerrainTile]) -> viser.ViserServer:
    server = viser.ViserServer(host=cfg.host, port=cfg.port, label="PIE Terrain Gallery")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.gui.add_markdown(
        "## PIE terrain gallery\n"
        "Columns: terrain types. Rows: levels **0–9**. Units: metres.\n\n"
        "**Green**: spawn; **orange**: route points. "
        "Select a type/level to inspect it closely.\n\n"
        "Drag to orbit; right-drag to pan; scroll to zoom. "
        "After editing the terrain code, restart this script."
    )
    terrain_filter = server.gui.add_dropdown(
        "Terrain", ("all", *PIE_PARKOUR_TERRAINS_CFG.sub_terrains), initial_value="all",
    )
    level_filter = server.gui.add_dropdown("Level", ("all", *(str(i) for i in range(10))), initial_value="all")
    show_labels = server.gui.add_checkbox("Terrain labels", initial_value=True)
    show_route = server.gui.add_checkbox("Route and spawn", initial_value=True)
    show_numbers = server.gui.add_checkbox("Waypoint numbers", initial_value=False)
    focus = server.gui.add_button("Focus visible terrains")
    overview = server.gui.add_button("Show all terrains")
    info = server.gui.add_markdown("")
    handles = []

    for tile in tiles:
        root = f"/terrains/{tile.name}/{tile.level}"
        frame = server.scene.add_frame(root, position=tile.offset, show_axes=False)
        server.scene.add_mesh_trimesh(f"{root}/geometry", tile.mesh)
        length, width = tile.size
        corners = np.array([[0, 0, 0.02], [length, 0, 0.02],
                            [length, width, 0.02], [0, width, 0.02], [0, 0, 0.02]])
        server.scene.add_line_segments(
            f"{root}/boundary", np.stack((corners[:-1], corners[1:]), axis=1),
            colors=(100, 130, 160), thickness=0.025,
        )
        label = server.scene.add_label(
            f"{root}/label", f"{tile.name} | L{tile.level} | d={tile.level / 9:.2f}",
            position=(length / 2, -0.45, 0.1), anchor="top-center", font_screen_scale=0.7,
        )
        route_frame = server.scene.add_frame(f"{root}/route", show_axes=False)
        points = np.vstack((tile.origin, tile.goals)) + [0, 0, 0.12]
        server.scene.add_line_segments(
            f"{root}/route/line", np.stack((points[:-1], points[1:]), axis=1),
            colors=(255, 175, 40), thickness=0.025,
        )
        server.scene.add_point_cloud(
            f"{root}/route/goals", points=points[1:].astype(np.float32),
            colors=(255, 175, 40), point_size=0.17, point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"{root}/route/spawn", points=points[:1].astype(np.float32),
            colors=(50, 220, 100), point_size=0.25, point_shape="circle",
        )
        numbers = server.scene.add_frame(f"{root}/route/numbers", show_axes=False, visible=False)
        for index, point in enumerate(points):
            server.scene.add_label(
                f"{root}/route/numbers/{index}", "spawn" if index == 0 else str(index),
                position=point + [0, 0, 0.18], font_screen_scale=0.7,
            )
        handles.append((tile, frame, label, route_frame, numbers))

    def visible_tiles():
        return [tile for tile in tiles
                if terrain_filter.value in ("all", tile.name)
                and level_filter.value in ("all", str(tile.level))]

    def focus_camera(client):
        selected = visible_tiles()
        bounds = np.stack([tile.mesh.bounds + tile.offset for tile in selected])
        low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
        center = (low + high) / 2
        span = max(float(np.max(high - low)), 6.0)
        client.camera.up_direction = (0, 0, 1)
        client.camera.position = center + np.array([0.15, -0.8, 0.9]) * span
        client.camera.look_at = center

    def update_visibility(_event=None):
        with server.atomic():
            for tile, frame, label, route_frame, numbers in handles:
                frame.visible = (terrain_filter.value in ("all", tile.name)
                                 and level_filter.value in ("all", str(tile.level)))
                label.visible = show_labels.value
                route_frame.visible = show_route.value
                numbers.visible = show_numbers.value
            info.content = (
                f"**{len(visible_tiles())} / {len(tiles)} tiles** · seed {cfg.seed}\n\n"
                "Difficulty = level / 9, matching `play_pie.py`. "
                "Training samples within difficulty bands; individual layouts vary."
            )

    def filter_changed(event):
        update_visibility(event)
        for client in server.get_clients().values():
            focus_camera(client)

    terrain_filter.on_update(filter_changed)
    level_filter.on_update(filter_changed)
    for control in (show_labels, show_route, show_numbers):
        control.on_update(update_visibility)

    @focus.on_click
    def _focus(event):
        if event.client is not None:
            focus_camera(event.client)

    @overview.on_click
    def _overview(event):
        terrain_filter.value = "all"
        level_filter.value = "all"
        filter_changed(event)

    server.on_client_connect(focus_camera)
    update_visibility()
    return server


def main(cfg: PreviewConfig) -> None:
    tiles = build_tiles(cfg)
    print(f"Built {len(tiles)} tiles: {len(PIE_PARKOUR_TERRAINS_CFG.sub_terrains)} types x 10 levels (0–9).")
    if cfg.check:
        for name in PIE_PARKOUR_TERRAINS_CFG.sub_terrains:
            selected = [tile for tile in tiles if tile.name == name]
            print(f"  {name}: levels {[tile.level for tile in selected]}")
        return
    server = build_viewer(cfg, tiles)
    print(f"Open http://{cfg.host}:{server.get_port()} — Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main(tyro.cli(PreviewConfig))
