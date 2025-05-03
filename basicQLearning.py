import numpy as np
import gymnasium as gym


class QLearningAgent:
    def __init__(self, environment, N=100, gamma=0.9, test_episodes=10, iterations=1000):
        self.gamma = gamma
        self.iterations = iterations
        self.test_episodes = test_episodes
        self.N = N
        self.environment = environment
        self.reward_dict = dict()
        self.transition_dict = dict()
        self.value_dict = dict()
        self.init_value_dict()
        return

    def init_value_dict(self):
        for x in range(self.environment.observation_space.n):
            self.value_dict[x] = 0
        for i in range(self.environment.action_space.n):
            for j in range(self.environment.observation_space.n):
                self.transition_dict[(j,i)] = dict()
        return

    def make_n_moves(self):
        obs_curr, _ = self.environment.reset()
        for i in range(self.N):
            action = self.environment.action_space.sample()
            obs_next, reward, terminated, truncated, _ = self.environment.step(action)
            self.reward_dict[(obs_curr, action, obs_next)] = reward
            if (obs_curr, action) not in self.transition_dict:
                self.transition_dict[(obs_curr, action)] = dict()
            if obs_next not in self.transition_dict[(obs_curr, action)]:
                self.transition_dict[(obs_curr, action)][obs_next] = 0
            self.transition_dict[(obs_curr, action)][obs_next] += 1
            obs_curr = obs_next
            if terminated or truncated:
                obs_curr, _ = self.environment.reset()
        return

    def calc_Q_value(self, state, action):
        Q_value = 0
        target_states = self.transition_dict[(state, action)]
        total = sum(target_states.values()) if len(target_states) > 0 else 0.0
        for t_state in target_states:
            p = target_states[t_state] / total
            Q_value += p * (self.reward_dict[(state, action, t_state)] + self.gamma * self.value_dict[t_state])
        return Q_value

    def choose_best_move(self, state):
        best_action, best_q_value = None, None
        for i in range(self.environment.action_space.n):
            Q_value = self.calc_Q_value(state, i)
            if best_q_value is None or Q_value > best_q_value:
                best_q_value = Q_value
                best_action = i
        if best_action is None:
            best_action = self.environment.action_space.sample()
        return best_action

    def episode(self):
        obs_curr, _ = self.environment.reset()
        total_reward = 0.0
        while True:
            action = self.choose_best_move(obs_curr)
            obs_next, reward, terminated, truncated, _ = self.environment.step(action)
            self.reward_dict[(obs_curr, action, obs_next)] = reward
            if (obs_curr, action) not in self.transition_dict:
                self.transition_dict[(obs_curr, action)] = dict()
            if obs_next not in self.transition_dict[(obs_curr, action)]:
                self.transition_dict[(obs_curr, action)][obs_next] = 0
            self.transition_dict[(obs_curr, action)][obs_next] += 1
            total_reward += reward
            if terminated or truncated:
                break
            obs_curr = obs_next
        return total_reward

    def calc_values(self):
        for i in range(self.environment.observation_space.n):
            state_vals = [self.calc_Q_value(i, j) for j in range(self.environment.action_space.n)]
            max_val = max(state_vals)
            self.value_dict[i] = max_val
        return

    def train(self):
        for i in range(self.iterations):
            self.make_n_moves()
            self.calc_values()
            total_r = 0.0
            for j in range(self.test_episodes):
                total_r += self.episode()
            total_r /= self.test_episodes
            print("Iteration "+str(i)+":", total_r)
            if total_r >= 0.8:
                print("Solved!")
                break
        return

    def print_dicts(self):
        print(self.reward_dict)
        print(self.transition_dict)
        return

# main
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode=None, map_name="8x8", is_slippery=True)
    q = QLearningAgent(env)
    q.train()