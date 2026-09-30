"""Scene obstacle helpers for data collection demos.

Register rigid objects on ``env_cfg.scene`` so you can keep DataCollection_* scripts
short and add new assets here only.
"""

from __future__ import annotations

import os
import shutil


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".bmp", ".tga")


def _local_texture_map(usd_dir: str) -> dict[str, str]:
    """Basename (lower) -> absolute POSIX path for every image under textures/."""
    found: dict[str, str] = {}
    for subdir in ("textures", os.path.join("Props", "textures")):
        tex_dir = os.path.join(usd_dir, subdir)
        if not os.path.isdir(tex_dir):
            continue
        for name in os.listdir(tex_dir):
            if name.startswith("_"):
                continue
            if os.path.splitext(name)[1].lower() not in _IMAGE_EXTS:
                continue
            found[name.lower()] = os.path.abspath(os.path.join(tex_dir, name)).replace("\\", "/")
    return found


def _shader_id(prim) -> str:
    attr = prim.GetAttribute("info:id")
    if attr and attr.Get():
        return str(attr.Get())
    return ""


def _rewire_usd_textures_to_local(usd_path: str, usd_dir: str) -> None:
    """Rewrite existing texture file attrs to local ``textures/`` images.

    Only touches attributes that already exist. Does not CreateInput / Define
    prims (that expires referenced MeshConverter prims and crashes Isaac).
    """
    from pxr import Sdf, Usd, UsdShade

    if not os.path.isfile(usd_path):
        return

    textures = _local_texture_map(usd_dir)
    if not textures:
        print(f"[WARN] No local textures under {usd_dir}/textures")
        return
    default_tex = textures.get("blue_bin_texture0.png") or next(iter(textures.values()))

    stage = Usd.Stage.Open(usd_path)
    jobs: list[tuple[str, str, str]] = []
    tex_paths: list[str] = []
    preview_paths: list[str] = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        sid = _shader_id(prim)
        name = prim.GetName()
        if "UsdUVTexture" in sid or name.endswith("Tex"):
            tex_paths.append(str(prim.GetPath()))
        if any(k in sid for k in ("PreviewSurface", "OmniPBR", "OmniSurface")):
            preview_paths.append(str(prim.GetPath()))
        for attr in prim.GetAttributes():
            if not attr.IsValid():
                continue
            try:
                val = attr.Get()
            except Exception:
                continue
            if not isinstance(val, Sdf.AssetPath):
                continue
            authored = (val.path or val.resolvedPath or "").replace("\\", "/")
            base = os.path.basename(authored).lower()
            ext = os.path.splitext(base)[1].lower()
            target = textures.get(base)
            if target is None and ext in _IMAGE_EXTS:
                target = default_tex
            if not target or authored == target:
                continue
            jobs.append((str(prim.GetPath()), attr.GetName(), target))

    n_files = 0
    for prim_path, attr_name, target in jobs:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            continue
        try:
            attr.Set(Sdf.AssetPath(target))
            n_files += 1
        except Exception as exc:
            print(f"[WARN] skip {prim_path}.{attr_name}: {exc}")

    n_connect = 0
    if tex_paths and preview_paths:
        tex_prim = stage.GetPrimAtPath(tex_paths[0])
        if tex_prim.IsValid():
            file_attr = tex_prim.GetAttribute("inputs:file")
            if file_attr and file_attr.IsValid():
                try:
                    file_attr.Set(Sdf.AssetPath(default_tex))
                    n_files += 1
                except Exception as exc:
                    print(f"[WARN] skip {tex_paths[0]}.inputs:file: {exc}")
            tex = UsdShade.Shader(tex_prim)
            rgb = tex.GetOutput("rgb") if tex else None
            if rgb:
                for ppath in preview_paths:
                    pprim = stage.GetPrimAtPath(ppath)
                    if not pprim.IsValid():
                        continue
                    preview = UsdShade.Shader(pprim)
                    inp = preview.GetInput("diffuseColor") if preview else None
                    if not inp:
                        continue
                    iattr = inp.GetAttr()
                    if not iattr or not iattr.IsValid():
                        continue
                    try:
                        inp.ConnectToSource(rgb)
                        n_connect += 1
                    except Exception as exc:
                        print(f"[WARN] skip connect {ppath}: {exc}")

    if n_files or n_connect:
        stage.GetRootLayer().Save()
    print(
        f"[INFO] Rewired textures in {os.path.basename(usd_path)}: "
        f"file_inputs={n_files}, reconnects={n_connect}, albedo={default_tex}"
    )


def _ensure_props_texture_copy(usd_dir: str) -> None:
    """Copy textures next to Props/*.usd so leftover relative paths still resolve."""
    src_dir = os.path.join(usd_dir, "textures")
    dst_dir = os.path.join(usd_dir, "Props", "textures")
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.startswith("_"):
            continue
        if os.path.splitext(name)[1].lower() not in _IMAGE_EXTS:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)


def _prepare_mesh_converter(glb_path: str, usd_dir: str, **converter_kwargs):
    """Run MeshConverter and repair stale texture references in the cached USD."""
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=glb_path,
            usd_dir=usd_dir,
            force_usd_conversion=False,
            **converter_kwargs,
        )
    )
    _ensure_props_texture_copy(usd_dir)
    _rewire_usd_textures_to_local(converter.usd_path, usd_dir)
    props_usd = os.path.join(usd_dir, "Props", "instanceable_meshes.usd")
    _rewire_usd_textures_to_local(props_usd, usd_dir)
    return converter


def add_obstacle_cube(env_cfg, pos, size, index: int) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    name = f"obstacle_cube_{index}"

    setattr(
        env_cfg.scene,
        name,
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        ),
    )


def blue_bin_texture_path() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../scene_new/_converted_blue_bin/textures/blue_bin_texture0.png",
        )
    )


def add_bin_fill_light(xy, z: float = 1.9, intensity: float = 4000.0) -> None:
    """Small indoor point light above the mesh bin (not the outdoor HDR sky).

    GS lab appearance is baked. The spawned bin is a PBR mesh: with skyLight
    off it has no irradiance and renders black even if the PNG is bound.
    """
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    path = "/World/binFillLight"
    if stage.GetPrimAtPath(path):
        return
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateRadiusAttr(0.15)
    light.CreateColorAttr((1.0, 1.0, 1.0))
    light.CreateTreatAsPointAttr(True)
    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3f(float(xy[0]), float(xy[1]), float(z)))
    print(
        f"[INFO] Local bin fill light at ({xy[0]:.2f}, {xy[1]:.2f}, {z:.2f}), "
        f"intensity={intensity} (outdoor skyLight still off)"
    )


def apply_blue_bin_visual(
    prim_path: str = "/World/envs/env_0/blue_bin_0",
    *,
    texture_path: str | None = None,
) -> str:
    """Bind OmniPBR + the local albedo PNG onto the spawned mesh bin.

    MeshConverter writes ``textures/blue_bin_texture0.png`` next to the USD but
    does not author a UsdUVTexture graph, so the live prim renders black even
    with a fill light. This overrides that after the env has spawned.
    """
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf, Usd

    from isaaclab.sim.utils import bind_visual_material, safe_set_attribute_on_usd_prim

    if texture_path is None:
        texture_path = blue_bin_texture_path()
    texture_path = os.path.abspath(texture_path)
    if not os.path.isfile(texture_path):
        raise FileNotFoundError(f"Blue bin texture not found: {texture_path}")

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"No blue bin prim at {prim_path}")

    mtl_prim = "/World/Looks/blue_bin_albedo"
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
        raise RuntimeError("OmniPBR Shader missing for blue bin")

    tex_attr = shader.GetAttribute("inputs:diffuse_texture")
    if not tex_attr.IsValid():
        tex_attr = shader.CreateAttribute("inputs:diffuse_texture", Sdf.ValueTypeNames.Asset)
    tex_attr.Set(Sdf.AssetPath(texture_path.replace("\\", "/")))
    safe_set_attribute_on_usd_prim(
        shader, "inputs:diffuse_color_constant", (0.12, 0.38, 0.78), camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:reflection_roughness_constant", 0.72, camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:metallic_constant", 0.0, camel_case=False
    )
    safe_set_attribute_on_usd_prim(
        shader, "inputs:albedo_brightness", 1.35, camel_case=False
    )

    bound = [prim_path]
    bind_visual_material(prim_path, mtl_prim, stronger_than_descendants=True)
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        path = str(prim.GetPath())
        bind_visual_material(path, mtl_prim, stronger_than_descendants=True)
        bound.append(path)

    print(
        f"[INFO] Blue bin albedo bound ({os.path.basename(texture_path)}) -> {bound}"
    )
    return mtl_prim


def add_blue_bin(env_cfg, pos, index: int) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas import schemas_cfg

    name = f"blue_bin_{index}"

    glb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/blue_bin.glb")
    )

    usd_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/_converted_blue_bin")
    )

    converter = _prepare_mesh_converter(
        glb_path,
        usd_dir,
        make_instanceable=False,  # must be False for RayCaster LiDAR to find Mesh prims
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    setattr(
        env_cfg.scene,
        name,
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=converter.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                scale=(0.65, 0.65, 0.65),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.5, 0.5, 0.5, 0.5)),
        ),
    )


def add_table(env_cfg, pos, index: int) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas import schemas_cfg

    name = f"table_{index}"

    glb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/table.glb")
    )

    usd_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/_converted_table1")
    )

    converter = _prepare_mesh_converter(
        glb_path,
        usd_dir,
        make_instanceable=False,  # must be False for RayCaster LiDAR to find Mesh prims
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    setattr(
        env_cfg.scene,
        name,
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=converter.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                scale=(1.25, 1.25, 1.25),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.5, 0.5, 0.5, 0.5)),
        ),
    )


def add_chair(env_cfg, pos, index: int) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas import schemas_cfg

    name = f"chair_{index}"

    glb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/chair.glb")
    )

    usd_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scene_new/_converted_chair")
    )

    converter = _prepare_mesh_converter(
        glb_path,
        usd_dir,
        make_instanceable=False,  # must be False for RayCaster LiDAR to find Mesh prims
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    setattr(
        env_cfg.scene,
        name,
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=converter.usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
                scale=(1, 1, 1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.5, 0.5, -0.5, -0.5)),
        ),
    )
