from ale_py import ALEInterface
from ale_py.roms import Pong
import numpy as np
from datetime import datetime
import time
import psutil
import os
import datetime

start_time = time.time()
datetime_start_time = datetime.datetime.now()

ale = ALEInterface()
ale.loadROM(Pong)
import gym

env = gym.make('ALE/Pong-v5')
env.unwrapped.get_action_meanings()
env.action_space
env.observation_space
observation = env.reset()
cumulated_reward = 0

frames = []
for t in range(1000):
    #     print(observation)
    frames.append(env.render(mode='rgb_array'))
    # very stupid agent, just makes a random action within the allowd action space
    action = env.action_space.sample()
    #     print("Action: {}".format(t+1))
    observation, reward, done, info = env.step(action)
    #     print(reward)
    cumulated_reward += reward
    if done:
        print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
        break
print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))

env.close()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(epx, eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


def model_step(model, observation, prev_x):
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!

    return action, prev_x


def play_game(env, model):
    observation = env.reset()

    frames = []
    cumulated_reward = 0

    prev_x = None  # used in computing the difference frame

    for t in range(1000):
        frames.append(env.render(mode='rgb_array'))
        action, prev_x = model_step(model, observation, prev_x)
        observation, reward, done, info = env.step(action)
        cumulated_reward += reward
        if done:
            print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
            break
    print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))
    env.close()



def train_model(env, model, total_episodes=100):
    hist = []
    observation = env.reset()

    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(epx, eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            hist.append((episode_number, reward_sum, running_reward))
            end_time = time.time()
            end_time_CPU = time.process_time()
            datetime_end_time = datetime.datetime.now()
            execution_time = end_time - start_time
            execution_time_CPU = end_time_CPU - start_time
            datetime_execution_time = datetime_end_time - datetime_start_time
            pid = os.getpid()
            python_process = psutil.Process(pid)
            memoryUse = python_process.memory_info()[0] / 2. ** 30
            CPU_core_usage = psutil.cpu_percent()
            print('resetting env. episode %f, reward total was %f. running mean: %f' % (
            episode_number, reward_sum, running_reward))
            # print("Start Time:", datetime_start_time)
            # print("End Time:", datetime_end_time)
            print(f" Execution Time: {datetime_execution_time}")
            print('CPU being used in Percentage:', CPU_core_usage)
            print('Ram Statics:', psutil.virtual_memory())  # physical memory usage
            # print('Memory used in Percentage:', psutil.virtual_memory()[2])
            f = open("Ananlysis.txt", "a")
            f.write('resetting env. episode %f, reward total was %f. running mean: %f' % (
                episode_number, reward_sum, running_reward))
            f.write(f"CPU being used in Percentage: {CPU_core_usage}")
            f.write(f"Ram Statics: {psutil.virtual_memory()}")
            f.write(f" Execution Time: {datetime_execution_time}")
            if running_reward >= -15:
                print("Start Time:", datetime_start_time)
                print("End Time:", datetime_end_time)
                break
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None
            if episode_number == total_episodes:
                return hist

        # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        #   print('ep {}: game finished, reward: {}'.format(episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))


# model initialization
H = 200  # number of hidden layer neurons
D = 80 * 80  # input dimensionality: 80x80 grid
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)

# import pickle
# model = pickle.load(open('model.pkl', 'rb'))
# hyperparameters
batch_size = 10  # every how many episodes to do a param update?
# learning_rate = 1e-4
learning_rate = 1e-2

gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}
# rmsprop memory
hist1 = train_model(env, model, total_episodes=100000)
f = open("Ananlysis.txt", "a")
f.close()
