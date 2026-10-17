import gymnasium as gym

class PongActionWrapper(gym.ActionWrapper):
    """
    Pong has 6 actions but only 3 matter:
        0 NOOP  →  0
        2 RIGHT (paddle up)   →  1
        3 LEFT  (paddle down) →  2

    Action IDs 1 (FIRE), 4 (RIGHTFIRE), 5 (LEFTFIRE) are functionally
    identical to NOOP / RIGHT / LEFT in Pong.

    Reducing from 6 → 3 means:
      - ε-greedy picks a useful action 2× more often
      - Q-network doesn't waste capacity on duplicate outputs
      - Policy converges faster
    """
    # Mapping: reduced index → original ALE action
    _ACTIONS = [0, 2, 3]   # NOOP, UP, DOWN

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self._ACTIONS))

    def action(self, act):
        return self._ACTIONS[act]