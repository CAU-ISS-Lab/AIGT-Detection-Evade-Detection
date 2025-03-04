import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from framework import RLADmodel,RLADloss
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from dataloader import get_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import argparse
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(30)

class State():
    def __init__(self):
        left=random.uniform(-1,0)
        right=random.uniform(0,1)
        self.distribution=[left,right]
        self.train_loss=0

    def reset(self):
        left = random.uniform(-1, 1)
        right = random.uniform(0, 1)
        self.distribution = [left, right]
        self.train_loss = 0
        return self

    def get_dis(self):
        return self.distribution

    def get_train_loss(self):
        return self.train_loss

    def set_dis(self,left,right):
        self.distribution=[left,right]

    def set_loss(self,train_loss):
        self.train_loss=train_loss

class Env():
    def __init__(self,criterion,device,threshold):
        self.device=device
        self.criterion=criterion
        self.threshold=threshold

    def reset(self):
        state=State()
        return state

    def step(self,feature,aug_feature,targets):
        reward=0.0
        loss, outputs = self.criterion(feature, aug_feature, targets)
        # print(outputs)
        difference=loss.item()-self.threshold
        if difference>0:
            reward+=-np.log(difference+1)
        else:
            reward+=np.exp(-difference)
        return reward,loss



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,noise):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.action = nn.Linear(32, action_dim)
        self.noise=noise

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.action(x)
        return action

    def sample(self, state,t):
        action = self.forward(state)
        deta_mean=action[:,0]
        deta_std=action[:,1]
        std=1
        std=std-((t+1)/50)*0.13
        mean_std=2
        mean_std=mean_std-((t+1)/50)*0.28
        if (t+1)%50==0:
            print(t,std,mean_std)
        normal = torch.distributions.Normal(deta_mean, mean_std)
        action_mean = normal.rsample()
        action_mean = torch.tanh(action_mean)
        normal = torch.distributions.Normal(deta_std, std)
        action_std = normal.rsample()
        action_std = torch.tanh(action_std)
        tmp=torch.zeros_like(action_std)
        ones=torch.ones(state[:,1].shape)
        bias_value=random.random()*0.2
        bias = torch.ones_like(action_std) * bias_value
        if self.noise == "gaussian" and not torch.all(torch.gt(action_std + state[:, 1], tmp)):
            action_std=torch.clamp(action_std,bias-state[:,1],ones-bias)
            # print(action_std,state[:,1])
        else:
            if not torch.all(torch.gt(action_std+ state[:, 1],action_mean+state[:,0])):
                action_std=torch.clamp(action_mean+state[:,0]-state[:,1]+bias,ones)
        action=torch.cat((action_mean.unsqueeze(1),action_std.unsqueeze(1)),dim=-1)

        return action

    def evaluate(self, state, epsilon=1e-6):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        # print(state)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG():
    def __init__(self, state_dim, action_dim, buffer_size,learning_rate,encoder,training_lr,device,noise):
        self.actor = Actor(state_dim, action_dim,noise)
        self.actor_target = Actor(state_dim, action_dim,noise)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        self.encoder=encoder
        self.encoder_optimizer=optim.Adam(self.encoder.parameters(),lr=training_lr)
        self.device=device
        self.encoder=self.encoder.to(self.device)
        self.encoder_scheduler=torch.optim.lr_scheduler.StepLR(self.encoder_optimizer,50,0.97)
        self.critic_scheduler=torch.optim.lr_scheduler.StepLR(self.critic_optimizer, 50, 0.98)
        self.actor_scheduler=torch.optim.lr_scheduler.StepLR(self.actor_optimizer, 50, 0.98)

    def get_action(self, state,episode):
        state = torch.FloatTensor(state.get_dis())
        action= self.actor.sample(state.unsqueeze(0),episode)
        return action.detach().numpy().squeeze()

    def get_feature(self,distribution,inputs):
        feature, aug_feature = self.encoder(inputs,  distribution)
        return feature,aug_feature

    def encoder_train(self,batch_loss):
        self.encoder_optimizer.zero_grad()
        batch_loss.backward()
        self.encoder_optimizer.step()

    def update_lr(self):
        self.encoder_scheduler.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def actor_critic_train(self, batch_size,episode):
        if self.buffer.size < batch_size:
            return

        states, actions, rewards, next_states, terminals = self.buffer.sample(batch_size)
        state_list=[]
        next_state_list=[]
        for tmp in states:
            state_list.append(tmp.get_dis())
        state = torch.FloatTensor(state_list)
        for tmp in next_states:
            next_state_list.append(tmp.get_dis())
        state = torch.FloatTensor(state_list)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor(next_state_list)
        terminals = torch.FloatTensor(terminals)
        next_action= self.actor_target.sample(next_state,episode)
        target_q = self.critic_target(next_state, next_action)
        target_q = rewards + (1 - terminals) * 0.99 * target_q
        q = self.critic(state, actions)
        print(q.shape,'target:',target_q.shape)
        critic_loss = torch.mean((q - target_q) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions = self.actor(state)
        q = self.critic(state, new_actions)
        actor_loss = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()
        if (episode+1)%50==0:
            print("actor_Lr:{}".format(self.actor_optimizer.state_dict()['param_groups'][0]['lr']))
            print("encoder_Lr:{}".format(self.encoder_optimizer.state_dict()['param_groups'][0]['lr']))

    def update_target_networks(self):
        tau = 0.001
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_encoder(self,path):
        torch.save(self.encoder.state_dict(),path)

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.state_buffer = np.empty(buffer_size,dtype=object)
        self.next_state_buffer = np.empty(buffer_size,dtype=object)
        self.action_buffer = np.zeros((buffer_size, action_dim))
        self.reward_buffer = np.zeros((buffer_size, 1))
        self.done_buffer = np.zeros((buffer_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices]
        )

def main(args):
    state_dim = args.state_dim  # State dimension representing mean and std
    action_dim = args.action_dim # Action dimension representing mean and std changes
    buffer_size = args.buffer_size # Replay buffer size
    batch_size = args.batch_size
    # rank, world_size = setup_distributed()
    device = args.device
    weight_1=args.weight_1
    weight_2=args.weight_2
    weight_3=args.weight_3
    episode_num=args.episode_num
    step_num=args.step_num
    lr=args.learning_rate
    threshold=args.threshold
    train_lr=args.encoder_lr
    if device is None:
        device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    state = State()
    source = args.data_path
    train_dataset = get_data(source)
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    pre_model = AutoModel.from_pretrained(args.PLM)
    pre_model.requires_grad_(False)
    model = RLADmodel(tokenizer, pre_model, device, args.is_aug)
    criterion = RLADloss(weight_1, weight_2, weight_3)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    env = Env(criterion, device, threshold)
    agent = DDPG(state_dim, action_dim, buffer_size, lr,model,train_lr,device,args.is_aug)
    # Training loop
    final_state = []
    epoch_reward = []
    for episode in range(episode_num):
        state=env.reset()
        total_reward = 0
        for step in range(step_num):
            next_state=State()
            now_dis = state.get_dis()
            action = agent.get_action(state,episode)
            next_dis=[x + y for x, y in zip(now_dis, action)]
            next_state.set_dis(next_dis[0],next_dis[1])
            step_loss=0
            step_reward=0
            iter=0
            with tqdm(train_loader) as loop:
                for sample_batch in loop:
                    data=sample_batch['text']
                    targets = sample_batch['label'].to(device)
                    feature,aug_feature=agent.get_feature(next_dis,data)
                    batch_reward,batch_loss=env.step(feature,aug_feature,targets)
                    step_reward+=10*batch_reward
                    step_loss+=batch_loss.item()
                    iter+=1
                    loop.set_postfix(batch_reward=10*batch_reward,batch_loss=batch_loss.item())
                    agent.encoder_train(batch_loss)
            step_loss=step_loss/iter
            step_reward=step_reward/iter
            state.set_loss=step_loss
            done = False
            agent.buffer.add(state, action, step_reward, next_state, done)
            agent.actor_critic_train(batch_size=1,episode=episode)
            state = next_state
            total_reward += step_reward

            if done:
                break
        agent.update_lr()
        epoch_reward.append(total_reward)
        final_state.append(state.get_dis())
        if (episode + 1) % 5 == 0:
            print(epoch_reward)
            y = [i for i in range(1, episode + 2)]
            print(y)
            plt.plot(y, epoch_reward)
            plt.savefig('reward.png')
            plt.show()
        print(f"Episode: {episode}, Total Reward: {total_reward},state: {state.get_dis()}")
        if (episode + 1) % 20 == 0:
            reward_array = np.array(epoch_reward, dtype='float32')
            np.save(args.reward_save_path, reward_array)
            agent.save_encoder(args.encoder_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='M4/arxiv_chatGPT_test.jsonl')
    parser.add_argument('--state_dim', type=int, default=2)
    parser.add_argument('--action_dim', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--PLM', type=str, default="roberta-base")
    parser.add_argument('--weight_1', type=float, default=0.5)
    parser.add_argument('--weight_2', type=float, default=1)
    parser.add_argument('--weight_3', type=float, default=0.01)
    parser.add_argument('--episode_num', type=int, default=300)
    parser.add_argument('--step_num', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--encoder_lr', type=float, default=0.0005)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--is_aug', type=str, default='gaussian')
    parser.add_argument('--reward_save_path', type=str, default='./reward/g_30_tmp.npz')
    parser.add_argument('--encoder_save_path',type=str,default='./encoder/g_30_new.pth')
    args, unparsed = parser.parse_known_args()
    main(args)






