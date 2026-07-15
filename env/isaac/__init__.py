"""Isaac Lab G1 wrappers (lazy imports to avoid pulling gym/gymnasium at init)."""

__all__ = ["IsaacG1Wrapper", "LatentHumanoidEnv"]


def __getattr__(name: str):
    if name == "IsaacG1Wrapper":
        from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper

        return IsaacG1Wrapper
    if name == "LatentHumanoidEnv":
        from env.isaac.latent_humanoid_env import LatentHumanoidEnv

        return LatentHumanoidEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
