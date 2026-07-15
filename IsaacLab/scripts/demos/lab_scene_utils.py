"""Load basement lab USD (3D Gaussian Splatting) and RayCaster mesh paths."""

from __future__ import annotations

import os

import numpy as np
from pxr import Gf, Usd, UsdGeom

LAB_EXTERNAL_PRIM = "/World/ExternalScene"
# lab.usda defaultPrim "World" is composed onto ExternalScene (no extra /World segment).
LAB_SCENE_RAYCAST_ROOT = LAB_EXTERNAL_PRIM
LAB_SCENE_REL = "../../scene_new/lab.usda"
GS_PAYLOAD_NAME = "3dgs_lab.usdz"

SENSOR_IMG_RES = (640, 480)
OUTPUT_IMG_RES = (480, 640)
IMG_RES_LANDSCAPE = OUTPUT_IMG_RES


def lab_usda_path(*, demos_dir: str | None = None) -> str:
    if demos_dir is None:
        demos_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(demos_dir, LAB_SCENE_REL))


def gs_payload_path(lab_path: str | None = None) -> str:
    if lab_path is None:
        lab_path = lab_usda_path()
    return os.path.join(os.path.dirname(lab_path), GS_PAYLOAD_NAME)


def _is_valid_prim(stage: Usd.Stage, path: str) -> bool:
    return stage.GetPrimAtPath(path).IsValid()


def discover_lab_raycast_roots(stage: Usd.Stage) -> tuple[str, ...]:
    """Return raycast search roots under ExternalScene that exist on the live stage."""
    if not _is_valid_prim(stage, LAB_EXTERNAL_PRIM):
        return ()

    candidates = (
        LAB_SCENE_RAYCAST_ROOT,
        f"{LAB_EXTERNAL_PRIM}/GroundPlane",
        f"{LAB_EXTERNAL_PRIM}/GroundPlane/CollisionMesh",
    )
    valid = [p for p in candidates if _is_valid_prim(stage, p)]
    if valid:
        return (valid[0],)

    # Fallback: any Mesh under ExternalScene.
    ext = stage.GetPrimAtPath(LAB_EXTERNAL_PRIM)
    for prim in Usd.PrimRange(ext):
        if prim.GetTypeName() == "Mesh":
            parent = str(prim.GetPath().GetParentPath())
            return (parent if _is_valid_prim(stage, parent) else LAB_SCENE_RAYCAST_ROOT,)
    return (LAB_SCENE_RAYCAST_ROOT,) if _is_valid_prim(stage, LAB_SCENE_RAYCAST_ROOT) else ()


def default_obstacle_mesh_paths(
    env_prim_root: str,
    obstacle_names: tuple[str, ...] = ("blue_bin_0",),
) -> tuple[str, ...]:
    """RayCaster roots for spawned obstacles only (matches original DataCollection scripts)."""
    root = env_prim_root.rstrip("/")
    paths: list[str] = []
    for name in obstacle_names:
        name = name.strip()
        if not name:
            continue
        paths.append(name if name.startswith("/") else f"{root}/{name}")
    return tuple(paths)


def default_raycast_mesh_paths(
    env_prim_root: str,
    obstacle_names: tuple[str, ...] = ("blue_bin_0",),
    *,
    include_lab_scene: bool = False,
    stage: Usd.Stage | None = None,
) -> tuple[str, ...]:
    """Meshes for RayCaster. Lab scene is optional (depth_rgb only); never hardcode bad paths."""
    paths: list[str] = []
    if include_lab_scene:
        if stage is not None:
            paths.extend(discover_lab_raycast_roots(stage))
        elif _is_valid_prim(_get_stage(), LAB_SCENE_RAYCAST_ROOT):
            paths.append(LAB_SCENE_RAYCAST_ROOT)
    paths.extend(default_obstacle_mesh_paths(env_prim_root, obstacle_names))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def _get_stage() -> Usd.Stage:
    import omni.usd

    return omni.usd.get_context().get_stage()


def validate_gs_payload(lab_path: str | None = None) -> bool:
    gs_path = gs_payload_path(lab_path)
    return os.path.isfile(gs_path)


def load_lab_scene_usd(
    *,
    demos_dir: str | None = None,
    translate: tuple[float, float, float] = (2.0, -1.0, 1.85),
    rotate_z_deg: float = 50.0,
    remove_default_ground: bool = True,
    verbose: bool = True,
) -> str:
    """Reference lab.usda at /World/ExternalScene."""
    import omni.usd

    scene_path = lab_usda_path(demos_dir=demos_dir)
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Lab scene not found: {scene_path}")

    has_gs = validate_gs_payload(scene_path)
    if verbose:
        print(f"[INFO] Loading lab scene: {scene_path}")
        print(f"[INFO] ExternalScene prim: {LAB_EXTERNAL_PRIM}")
        if has_gs:
            print(f"[INFO] 3DGS payload: {gs_payload_path(scene_path)}")
        else:
            print(f"[WARN] 3DGS payload not found at {gs_payload_path(scene_path)}")

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(LAB_EXTERNAL_PRIM):
        prim = stage.DefinePrim(LAB_EXTERNAL_PRIM, "Xform")
        prim.GetReferences().AddReference(scene_path)

        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(*translate))
        xform.AddRotateZOp().Set(rotate_z_deg)
        xform.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    elif verbose:
        print("[INFO] Lab scene already loaded at ExternalScene")

    if remove_default_ground:
        ground_path = "/World/ground"
        if stage.GetPrimAtPath(ground_path):
            stage.RemovePrim(ground_path)

    if verbose:
        roots = discover_lab_raycast_roots(stage)
        if roots:
            print(f"[INFO] Lab RayCaster search root(s): {roots}")
        print(f"[INFO] ExternalScene valid: {stage.GetPrimAtPath(LAB_EXTERNAL_PRIM).IsValid()}")

    return scene_path


def rotate_sensor_ccw_to_landscape(img: np.ndarray) -> np.ndarray:
    """Portrait sensor frame → CCW 90° → 640×480 landscape (matches DataCollection_loop)."""
    return np.rot90(img, k=1)


def ensure_landscape_rgb(rgb: np.ndarray, img_res: tuple[int, int] | None = None) -> np.ndarray:
    """Alias kept for callers; always applies CCW 90° (img_res ignored)."""
    del img_res
    return rotate_sensor_ccw_to_landscape(rgb)
