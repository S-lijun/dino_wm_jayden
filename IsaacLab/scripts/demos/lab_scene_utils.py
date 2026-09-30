"""Load basement lab USD (3D Gaussian Splatting) and RayCaster mesh paths."""

from __future__ import annotations

import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

LAB_EXTERNAL_PRIM = "/World/ExternalScene"
# lab.usda defaultPrim "World" is composed onto ExternalScene (no extra /World segment).
LAB_SCENE_RAYCAST_ROOT = LAB_EXTERNAL_PRIM
# Collision proxies: hidden for GS RGB, must be visible for RTX camera depth.
LAB_DEPTH_PROXY_PATHS = (
    f"{LAB_EXTERNAL_PRIM}/GroundPlane",
    f"{LAB_EXTERNAL_PRIM}/Plane",
    f"{LAB_EXTERNAL_PRIM}/plane2",
    f"{LAB_EXTERNAL_PRIM}/plane3",
    f"{LAB_EXTERNAL_PRIM}/plane4",
)
LAB_SCENE_REL = "../../scene_new/lab.usda"
FLOOR_ONLY_REL = "../../scene_new/floor_only.usda"
FLOOR_TEXTURE_NAME = "lab_floor.jpg"
GS_PAYLOAD_NAME = "3dgs_lab.usdz"

SENSOR_IMG_RES = (640, 480)
OUTPUT_IMG_RES = (480, 640)
IMG_RES_LANDSCAPE = OUTPUT_IMG_RES


def lab_usda_path(*, demos_dir: str | None = None) -> str:
    if demos_dir is None:
        demos_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(demos_dir, LAB_SCENE_REL))


def floor_only_usda_path(*, demos_dir: str | None = None) -> str:
    if demos_dir is None:
        demos_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(demos_dir, FLOOR_ONLY_REL))


def floor_texture_path(*, demos_dir: str | None = None) -> str:
    usda = floor_only_usda_path(demos_dir=demos_dir)
    local = os.path.join(os.path.dirname(usda), FLOOR_TEXTURE_NAME)
    if os.path.isfile(local):
        return local
    fallback = os.path.abspath(os.path.join("C:\\gs_apt", "1000006893.jpg"))
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"Floor texture not found next to {usda} or at {fallback}"
    )


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


def _ensure_payloads_loaded(stage: Usd.Stage) -> None:
    ext = stage.GetPrimAtPath(LAB_EXTERNAL_PRIM)
    if ext and ext.IsValid():
        ext.Load()
    try:
        stage.Load()
    except Exception:
        pass


def _nurec_volume_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    prims: list[Usd.Prim] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsValid() or not prim.IsLoaded():
            continue
        flag = prim.GetAttribute("omni:nurec:isNuRecVolume")
        is_flag = bool(flag.IsValid() and flag.Get())
        if prim.GetTypeName() == "Volume" or is_flag:
            prims.append(prim)
    return prims


def _session_set_float4(prim: Usd.Prim, name: str, value: Gf.Vec4f) -> Usd.Attribute:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float4, custom=True)
    attr.Set(value)
    return attr


def apply_nurec_floor_color(
    *,
    deyellow: float = 0.55,
    verbose: bool = True,
) -> int:
    """Session-layer NuRec CCM on the Volume prim. Does not write assets.

    Official knob is OmniNuRecVolumeAPI on the UsdVol Volume, not the field
    child. Mixes blue into R/G so yellow goes grey instead of muddy brown.
    """
    stage = _get_stage()
    _ensure_payloads_loaded(stage)
    volumes = _nurec_volume_prims(stage)
    if not volumes:
        if verbose:
            print("[WARN] No NuRec Volume prim found; floor color override skipped.")
        return 0

    k = float(np.clip(deyellow, 0.0, 1.0))
    # R' = (1-k)R + k B, G' = (1-k)G + k B, B' = B. No blue boost.
    ccm_r = Gf.Vec4f(1.0 - k, 0.0, k, 0.0)
    ccm_g = Gf.Vec4f(0.0, 1.0 - k, k, 0.0)
    ccm_b = Gf.Vec4f(0.0, 0.0, 1.0, 0.0)

    session = stage.GetSessionLayer()

    def _apply(prim: Usd.Prim) -> None:
        try:
            prim.ApplyAPI("OmniNuRecVolumeAPI")
        except Exception as exc:
            if verbose:
                print(f"[WARN] ApplyAPI OmniNuRecVolumeAPI on {prim.GetPath()}: {exc}")
        for name, value in (
            ("omni:nurec:ccmR", ccm_r),
            ("omni:nurec:ccmG", ccm_g),
            ("omni:nurec:ccmB", ccm_b),
        ):
            attr = _session_set_float4(prim, name, value)
            got = attr.Get()
            if verbose:
                print(f"[INFO] set {prim.GetPath()}.{name} -> {got}")

    if session is not None:
        with Usd.EditContext(stage, session):
            for prim in volumes:
                _apply(prim)
    else:
        for prim in volumes:
            _apply(prim)

    if verbose:
        print(f"[INFO] NuRec floor CCM on {len(volumes)} Volume(s), deyellow={k:g}")
    return len(volumes)


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


def apply_photo_ground(
    *,
    ground_path: str = "/World/ground",
    texture_path: str | None = None,
    texture_scale: tuple[float, float] = (19.9, 20.8),
    demos_dir: str | None = None,
    verbose: bool = True,
) -> str:
    """Bind the saved lab carpet photo onto Isaac's default ground plane."""
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf, UsdGeom

    from isaaclab.sim.utils import bind_visual_material, safe_set_attribute_on_usd_prim

    if texture_path is None:
        texture_path = floor_texture_path(demos_dir=demos_dir)
    texture_path = os.path.abspath(texture_path)
    if not os.path.isfile(texture_path):
        raise FileNotFoundError(f"Floor texture not found: {texture_path}")

    mtl_prim = "/World/Looks/photo_floor"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(ground_path).IsValid():
        raise RuntimeError(f"No ground prim at {ground_path}")
    if stage.GetPrimAtPath(mtl_prim):
        stage.RemovePrim(mtl_prim)

    omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_path=mtl_prim,
        select_new_prim=False,
    )
    shader = stage.GetPrimAtPath(f"{mtl_prim}/Shader")
    if not shader.IsValid():
        raise RuntimeError("OmniPBR Shader missing")

    tex_attr = shader.GetAttribute("inputs:diffuse_texture")
    if not tex_attr.IsValid():
        tex_attr = shader.CreateAttribute("inputs:diffuse_texture", Sdf.ValueTypeNames.Asset)
    tex_attr.Set(Sdf.AssetPath(texture_path.replace("\\", "/")))
    safe_set_attribute_on_usd_prim(
        shader, "inputs:diffuse_color_constant", (1.0, 1.0, 1.0), camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:reflection_roughness_constant", 0.9, camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:metallic_constant", 0.0, camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:texture_scale", tuple(float(v) for v in texture_scale), camel_case=False
    )

    bind_visual_material(ground_path, mtl_prim, stronger_than_descendants=True)
    grid_look = stage.GetPrimAtPath(f"{ground_path}/Looks/theGrid")
    if grid_look.IsValid():
        UsdGeom.Imageable(grid_look).MakeInvisible()
    if verbose:
        print(f"[INFO] Photo ground: {texture_path} -> {ground_path} scale={texture_scale}")
    return texture_path


def set_lab_depth_proxies_visible(stage: Usd.Stage, visible: bool) -> list[str]:
    """Show/hide lab.usda collision meshes so RTX ``distance_to_image_plane`` hits the floor."""
    found: list[str] = []
    for path in LAB_DEPTH_PROXY_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        img = UsdGeom.Imageable(prim)
        if visible:
            img.MakeVisible()
        else:
            img.MakeInvisible()
        found.append(path)
    return found


def rotate_sensor_ccw_to_landscape(img: np.ndarray) -> np.ndarray:
    """Portrait sensor frame → CCW 90° → 640×480 landscape (matches DataCollection_loop)."""
    # rot90 returns a view with negative strides; copy so torch.from_numpy / imageio are safe.
    return np.ascontiguousarray(np.rot90(img, k=1))


def ensure_landscape_rgb(rgb: np.ndarray, img_res: tuple[int, int] | None = None) -> np.ndarray:
    """Alias kept for callers; always applies CCW 90° (img_res ignored)."""
    del img_res
    return rotate_sensor_ccw_to_landscape(rgb)
