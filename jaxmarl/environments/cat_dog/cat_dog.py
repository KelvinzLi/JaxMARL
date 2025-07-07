import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import chex
from jaxmarl.environments import spaces
from typing import Optional, Tuple

from flax import struct

from enum import IntEnum
from functools import partial

@struct.dataclass
class State:
    is_cat: bool
    light_on: bool
    barrier_removed: bool
    step: int

# P1:[Obs] x[Actions] = [Cat | Dog] x [0 | 1 | Show | Bail]
# P2:[Obs] x[Actions] = [Cat | Dog | 0 | 1] x [Cat | Dog | Bail]

class AliceActions(IntEnum):
    light_off = 0
    light_on = 1
    remove_barrier = 2
    bail_out = 3

class BobActions(IntEnum):
    cat = 0
    dog = 1
    bail_out = 2

class CatDog(object):

    barrier_removed_reward = -5

    correct_guess_reward = 10
    wrong_guess_reward = 10

    alice_bail_out_reward = 1
    bob_bail_out_reward = 0.5

    def __init__(self) -> None:
        super().__init__(num_agents=2)

        self.alice_action_set = jnp.array([
            AliceAction.light_off, AliceAction.light_on, AliceAction.remove_barrier, AliceAction.bail_out, 
        ])
        self.bob_action_set = jnp.array([
            BobAction.cat, BobAction.dog, BobAction.bail_out, 
        ])

        self.action_spaces = {
            "agent_0": spaces.Discrete(len(self.alice_action_set)), 
            "agent_1": spaces.Discrete(len(self.bob_action_set)), 
        }

        self.observation_spaces = {
            "agent_0": spaces.Box(low=0, high=1, shape=(2,)), 
            "agent_1": spaces.Box(low=0, high=1, shape=(4,)), 
        }

        self.agents = ["agent_0", "agent_1"]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        state = State(
            is_cat = (jax.random.uniform(key) > 0.5), 
            light_on = False, 
            barrier_removed = False, 
            step = 0, 
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        # light_on: bool
        # barrier_revealed: bool
        # step: int

        alice_action = actions["agent_0"]
        bob_action = actions["agent_1"]

        def alice_step_func(state):
            state = state.replace(
                light_on = (alice_action == AliceActions.light_on), 
                barrier_removed = (alice_action == AliceActions.barrier_removed), 
            )

            reward = 0
            reward = jax.cond(
                alice_action == AliceActions.barrier_removed, 
                lambda: barrier_removed_reward, 
                lambda: reward
            )
            reward = jax.cond(
                alice_action == AliceActions.bail_out, 
                lambda: alice_bail_out_reward, 
                lambda: reward
            )

            done = (alice_action == AliceActions.bail_out)

            return state, reward, done

        def bob_step_func(state):

            correct_guess_flag = jax.logical_or(
                jax.logical_and(bob_action == BobActions.cat, state.is_cat), 
                jax.logical_and(bob_action == BobActions.dog, jnp.logical_not(state.is_cat)), 
            )

            reward = 0
            reward = jax.cond(
                jax.logical_or(bob_action == BobActions.cat, bob_action == BobActions.dog), 
                jax.cond(
                    correct_guess_flag, 
                    lambda: correct_guess_reward, 
                    lambda: wrong_guess_reward, 
                ), 
                lambda: reward
            )
            reward = jax.cond(
                bob_action == BobActions.bail_out, 
                lambda: bob_bail_out_reward, 
                lambda: reward
            )

            done = True

            return state, reward, done

        state, reward, done = jax.cond(
            state.step == 0, 
            alice_step_func, 
            bob_step_func, 
            state
        )

        obs = self.get_obs(state)

        rewards = {agent_name: reward for agent_name in self.agents}
        dones = {agent_name: reward for agent_name in (*self.agents, "__all__")}
        infos = {}

        return obs, state, rewards, dones, infos
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        alice_obs = jnp.zeros((len(self.alice_obs_set),))
        alice_obs = jax.cond(
            state.is_cat,
            lambda: alice_obs.at[0].set(1), 
            lambda: alice_obs.at[1].set(1),
        )

        bob_obs = jnp.zeros((len(self.bob_obs_set),))
        bob_obs = jax.cond(
            state.barrier_removed,
            jax.cond(
                state.is_cat,
                lambda: bob_obs.at[0].set(1), 
                lambda: bob_obs.at[1].set(1),
            ), 
            lambda: bob_obs,
        )
        # TODO: should we indicate light bulb state even when barrier removed?
        bob_obs = jax.cond(
            state.light_on,
            lambda: bob_obs.at[2].set(1), 
            lambda: bob_obs.at[3].set(1),
        )

        return {"agent_0": alice_obs, "agent_1": bob_obs}

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        alice_flag = (state.step == 0)
        return {"agent_0": alice_flag, "agent_1": jnp.logical_not(alice_flag)}

    @property
    def name(self) -> str:
        return "CatDog"