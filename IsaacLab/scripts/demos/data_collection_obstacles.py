"""Scene obstacle helpers for data collection demos.

Register rigid objects on ``env_cfg.scene`` so you can keep DataCollection_* scripts
short and add new assets here only.
"""

from __future__ import annotations

import os


def _fix_stale_usd_texture_paths(usd_path: str, usd_dir: str) -> None:
    """Rewrite broken absolute texture paths to paths relative to ``usd_dir``.

    MeshConverter caches USD under ``usd_dir``; if the project was moved to another
    machine, baked texture paths may still point at the old absolute location.
    """
    from pxr import Sdf, Usd

    if not os.path.isfile(usd_path):
        return

    stage = Usd.Stage.Open(usd_path)
    modified = False

    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if not isinstance(val, Sdf.AssetPath):
                continue
            authored = val.path
            if not authored or os.path.isfile(authored):
                continue
            basename = os.path.basename(authored)
            for subdir in ("textures", os.path.join("Props", "textures")):
                candidate = os.path.join(usd_dir, subdir, basename)
                if not os.path.isfile(candidate):
                    continue
                rel = os.path.relpath(candidate, usd_dir).replace("\\", "/")
                if not rel.startswith("."):
                    rel = f"./{rel}"
                attr.Set(Sdf.AssetPath(rel))
                modified = True
                break

    if modified:
        stage.GetRootLayer().Save()


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
    _fix_stale_usd_texture_paths(converter.usd_path, usd_dir)
    props_usd = os.path.join(usd_dir, "Props", "instanceable_meshes.usd")
    _fix_stale_usd_texture_paths(props_usd, usd_dir)
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
