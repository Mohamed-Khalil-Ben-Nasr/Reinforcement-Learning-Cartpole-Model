from collections import namedtuple
import numpy as np
import gym
from tensorflow import keras
from keras import layers

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


def build_model(obs_size, HIDDEN_SIZE, n_actions):
    model = keras.Sequential([
        layers.Dense(HIDDEN_SIZE, input_shape=(obs_size,), activation="relu"),
        layers.Dense(n_actions, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["mae"])
    return model


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()[0]
    while True:
        # obs_v = torch.FloatTensor(obs) #turn to numpy
        obs = np.array(obs).reshape(1, -1)
        # act_probs_v = sm(net(obs_v)) #pass data to keras nn
        # act_probs = act_probs_v.data.numpy()
        act_probs = model(obs)
        act_probs = act_probs.numpy()
        act_probs /= act_probs.sum()
        action = np.random.choice(len(act_probs[0]), p=act_probs[0])
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs[0], action=action)
        episode_steps.append(step)
        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()[0]
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = [s.reward for s in batch]
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend([step.observation for step in steps])
        train_act.extend([step.action for step in steps])
    # basically replace map with [s.something for s in ss] (list comprehension)

    # train_obs_v = torch.FloatTensor(train_obs)
    train_obs = np.array(train_obs)
    # train_act_v = torch.LongTensor(train_act)
    train_act = np.array(train_act)
    return train_obs, train_act, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = build_model(obs_size, HIDDEN_SIZE, n_actions)
    # objective = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    for iter_no, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)):
        obs, acts, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # this where you train the model
        # optimizer.zero_grad()
        # action_scores_v = net(obs_v)
        # loss_v = objective(action_scores_v, acts_v)
        # loss_v.backward()
        # optimizer.step()
        # replace with fit
        history = model.fit(obs, acts, epochs=1, batch_size=BATCH_SIZE)
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, history.history['loss'][-1], reward_m, reward_b))
        # model.fit returns a history object (dictionary has keys like loss and accuracy)
        if reward_m > 199:
            print("Solved!")
            break
