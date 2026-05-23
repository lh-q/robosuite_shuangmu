import re
import matplotlib.pyplot as plt
import numpy as np

def parse_data_from_txt(file_path):
    '''
    return:
        steps, 
        rewards, 
        obs_data, 
        action_data

    '''
    steps = []
    rewards = []
    obs_data = []
    action_data = []

    with open(file_path, 'r') as file:
        content = file.read()

        # 匹配所有 Step
        step_matches = re.findall(r"Step\s+(\d+):", content)
        steps = list(map(int, step_matches))

        # 匹配所有 Rewards
        reward_matches = re.findall(r"Rewards:\s*([-\d.eE+]+)", content)
        rewards = list(map(float, reward_matches))

        # 匹配所有 Obs
        obs_matches = re.findall(r"Obs:\s*\[([\s\S]*?)\]", content, re.MULTILINE)
        for match in obs_matches:
            obs = list(map(float, match.replace('\n', '').split()))
            obs_data.append(obs)

        # 匹配所有 Action
        action_matches = re.findall(r"Action:\s*\[([\s\S]*?)\]", content, re.MULTILINE)
        for match in action_matches:
            action = list(map(float, match.replace('\n', '').split()))
            action_data.append(action)

    return steps, rewards, obs_data, action_data

