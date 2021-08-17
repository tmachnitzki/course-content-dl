from gym.wrappers import Monitor
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def wrap_env(env, path='./video'):
    env = Monitor(env, path , force=True)
    return env


def extract_information(model, test_env, save_folder,
                    n_repeats=100, save_video=False, save_csv=True):
    
    video_path = save_folder + 'video/'
    csv_path = save_folder + 'info/'
    Path(csv_path).mkdir(exist_ok=True, parents=True)
    
    test_env = wrap_env(test_env, path=video_path) if save_video else test_env
    for n_repeat in range(n_repeats):
        print('n_repeat: ', n_repeat)
        observation = test_env.reset()
        cum_reward = 0
        step = 0
        done = False
        this_rep_dfs = []
        while not done:
            
            x, y  = np.array(test_env.lander.position)
            action, states = model.predict(observation, deterministic=True)
            observation, reward, done, info = test_env.step(action)
            cum_reward += reward

            df = pd.DataFrame(dict(
                cum_reward=[cum_reward], step=[step],
                repeat=[n_repeat], x=[x], y=[y]
                ))
            
            this_rep_dfs.append(df)

            step += 1
        this_rep_dfs = pd.concat(this_rep_dfs)
        this_rep_dfs['last_reward'] = cum_reward
        this_rep_dfs.to_csv(f'{csv_path}_{n_repeat}.csv')
        
        print('total reward: ', cum_reward)

        
def performance_plot(df_info, fig=None):
    if fig is None:
        fig = plt.figure()
    sns.lineplot(x='repeat', y='last_reward', data=df_info.query('step==0'))

    rewards = df_info.query('step==0').last_reward

    plt.hlines(rewards.mean(), 0, 100, color='g',linestyles='dotted')

    ax = plt.gca()
    ax.set_xlabel('episodes')
    ax.set_ylabel('total reward')
    ax.set_title(f'Performance on random init x: M={rewards.mean():.2f}, Sd={rewards.std():.2f}')
    return fig

def movement_plot(df_info,fig=None, alpha=.8):
    if fig is None:
        fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([0,20])
    ax.set_ylim([0,15])
    for rep in df_info.repeat.unique():
        sns.scatterplot(x='x', y='y', data=df_info.query('repeat==@rep'), alpha=alpha)
    return fig
