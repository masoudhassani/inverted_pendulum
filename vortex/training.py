from environment import Environment

env = Environment()
current_state = env.reset()

for i in range(1000):
    action = env.sample()
    state, reward = env.step(action)
    print state