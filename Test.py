#################定义环境###########################
import numpy as np
start_point = np.array([0,0])
end_point = np.array([10,10])

class UAV:
    def __init__(self,position):
        super().__init__()  # 继承父类上的内容
        self.position = position

        #获取当前状态
    def get_state(self):
        return np.array(self.position)

    def take_action(self,action): #动作
        if action==0:
            self.position[0] = self.position[0] + 1

        if action==1:
            self.position[0] = self.position[0] - 1

        if action==2:
            self.position[1] = self.position[1] + 1

        if action==3:
            self.position[1] = self.position[1] - 1

    def reset(self,position):
        self.position[0] = 0
        self.position[1] = 0


    def get_reward(self): #奖励
        distance = np.linalg.norm(self.position - end_point) #用范数判断
        if distance == 0:
            reward = 100
            return reward

        else:
            return -distance


import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN算法类
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = []
        self.model = DQN(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    #定义epsilon贪心策略
    def epsilon_greedy_policy(self,state): #传入状态
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        else:
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    #记录经验
    def rember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def replay(self,batch_size):
        if len(self.memory) < batch_size: #如果经验回放数量不到batch_size，什么也不做
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([s[0] for s in batch])
        actions = torch.LongTensor([s[1] for s in batch])
        rewards = torch.FloatTensor([s[2] for s in batch])
        next_states = torch.FloatTensor([s[3] for s in batch])
        dones = torch.BoolTensor([s[4] for s in batch])

        q_values = self.model(states) #根据传入的状态产生Q值
        next_q_values = self.model(next_states) #下一个状态产生Q值
        max_next_q_values = torch.max(next_q_values, dim=1)[0] #下一个状态的最大Q值
        targets = rewards + self.gamma * max_next_q_values * (~dones) #贝尔曼最优方程

        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_values_for_actions, targets.detach())

        self.optimizer.zero_grad() #将所有参数的梯度清零，以避免上一次计算的梯度对本次计算造成影响
        loss.backward() #计算误差loss对神经网络参数的梯度,由神经网络的输出和训练数据的真实值计算得到
        self.optimizer.step() #更新神经网络参数

        # epsilon 贪心策略的 epsilon 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

start_point = np.array([0,0])
end_point = np.array([10,10])

#初始化DQN算法智能体
state_dim = 2
action_dim = 4
hidden_dim = 64
lr = 0.001
gamma = 0.95 #折减系数为0.95
epsilon = 0.95
epsilon_min = 0.01
epsilon_decay = 0.995
agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay)

#定义训练参数
batch_size = 32
episodes = 100
steps_per_episode = 100

for episode in range(episodes):
    # 初始化无人机位置
    uav = UAV(start_point)
    uav.reset(start_point)
    for step in range(steps_per_episode):
        # 获取当前状态
        state = uav.get_state()

        # 选择动作
        action = agent.epsilon_greedy_policy(state)

        # 执行动作
        uav.take_action(action)

        # 获取奖励
        reward = uav.get_reward()

        # 获取下一个状态
        next_state = uav.get_state()
        print(next_state)
        # 记录经验
        agent.rember(state, action, reward, next_state, False)

        # 更新神经网络
        agent.replay(batch_size)

        # 到达终点，停止本次训练
        if np.array_equal(uav.position, end_point):
            break
    print("Episode {}/{}: Steps = {}".format(episode + 1, episodes, step + 1),"无人机的坐标是:",state,"奖励为:",reward)
