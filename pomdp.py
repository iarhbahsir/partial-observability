import numpy as np


class PartiallyObservableEnv:
    NOISY = 0
    FAULTY = 1
    FULLY_OBS = 2

    # input: a gym environment
    def __init__(self, env, seed, noisy=None, faulty=None):
        self.env = env
        self.seed = seed
        np.random.seed(seed=self.seed)
        if not noisy and not faulty:
            self.po_type = PartiallyObservableEnv.FULLY_OBS
        elif noisy:
            self.po_type = PartiallyObservableEnv.NOISY
            self.mean = noisy[0]
            self.std = noisy[1]
        else:
            self.po_type = PartiallyObservableEnv.FAULTY
            self.percent_faulty = faulty[0]
            self.num_inputs_considered = faulty[1]
            faulty_mask_indices = np.random.choice(self.num_inputs_considered, int(np.floor(self.percent_faulty * 0.01 * self.num_inputs_considered)), replace=False)
            self.faulty_mask = np.ones(self.env.observation_space.shape[0])
            for index in faulty_mask_indices:
                self.faulty_mask[index] = 0

    def reset(self):
        return self.occlude(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.occlude(next_state), reward, done, info

    def render(self):
        return self.env.render()

    def occlude(self, state):
        if self.po_type == PartiallyObservableEnv.FULLY_OBS:
            return state
        elif self.po_type == PartiallyObservableEnv.NOISY:
            return np.maximum(np.minimum(state + np.random.normal(self.mean, self.std, np.shape(state)), self.env.observation_space.high), self.env.observation_space.low)
        else:
            return np.multiply(state, self.faulty_mask)
