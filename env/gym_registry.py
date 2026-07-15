"""Legacy OpenAI Gym env registration for dubins / pusht / wall / etc.

Import this module explicitly when you need ``gym.make("dubins")`` etc.
Isaac humanoid collection does NOT need it.
"""

from gym.envs.registration import register
import gym  # noqa: F401

register(
    id="dubins",
    entry_point="env.dubins.dubins_wrapper:DubinsWrapper",
    max_episode_steps=500,
)

register(
    id="maniskill",
    entry_point="env.maniskill.maniskill_wrapper:ManiskillWrapper",
    max_episode_steps=500,
)
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
