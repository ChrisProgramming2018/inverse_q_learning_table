import gym
from replay_buffer2 import ReplayBuffer 
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from utils import mkdir

class Agent():
    def __init__(self, state_size, action_size, config):
        self.action_size = action_size
        self.state_size = state_size
        self.Q = np.zeros([state_size, action_size])
        self.Q_inverse = np.zeros([state_size, action_size])
        self.debug_Q = np.zeros([state_size, action_size])
        self.Q_shift = np.zeros([state_size, action_size])
        self.r = np.zeros([state_size, action_size])  
        self.counter = np.zeros([state_size, action_size])
        self.gamma = config["gamma"]
        self.epsilon = 1
        self.lr = config["lr"]
        self.lr_iql_q = config["lr_iql_q"]
        self.lr_iql_r = config["lr_iql_r"]
        self.min_epsilon = config["min_epsilon"]
        self.max_epsilon =1
        self.episode = 15000
        self.decay = config["decay"]
        self.total_reward = 0
        self.eval_frq = 50
        self.render_env = False
        self.env = gym.make(config["env_name"])
        self.memory = ReplayBuffer((1,),(1,),config["buffer_size"], config["device"])
        self.gamma_iql = 0.99
        self.gamma_iql = 0.99
        self.lr_sh = config["lr_q_sh"]
        self.ratio = 1. / action_size
        self.eval_q_inverse = 50000
        self.episodes_qinverse = int(5e6)
        self.update_freq = config['freq_q']
        self.steps = 0
        pathname = "lr_inv_q {} lr_inv_r {} freq {}".format(self.lr_iql_q, self.lr_iql_r, self.update_freq)
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)
        tensorboard_name = str(config["locexp"]) + '/runs/' + "inverse" 
        self.writer_inverse = SummaryWriter(tensorboard_name)
        tensorboard_name = str(config["locexp"]) + '/runs/' + "expert" 
        self.writer_expert = SummaryWriter(tensorboard_name)
        self.last_100_reward_errors = deque(maxlen=100) 
        self.average_same_action = deque(maxlen=100) 
        self.expert_buffer_size = config["expert_buffer_size"]
    def act(self, state, epsilon, eval_pi=False, use_debug=False):

        if np.random.random() > epsilon or eval_pi:
            action = np.argmax(self.Q[state])
            if use_debug:
                action = np.argmax(self.debug_Q[state])
        else:
            action = self.env.action_space.sample() 
        return action
   
    def act_inverse_q(self, state):
        action = np.argmax(self.Q_inverse[state])
        return action
    
    def optimize(self, state, action, reward, next_state, debug=False):
        if debug:
            max_next_state = np.max(self.debug_Q[next_state])
            td_error =  max_next_state - self.debug_Q[state, action]
            self.debug_Q[(state,action)] = self.debug_Q[(state,action)] + self.lr * (reward + self.gamma *td_error)
            return

        max_next_state = np.max(self.Q[next_state])
        td_error =  max_next_state - self.Q[state, action]
        self.Q[(state,action)] = self.Q[(state,action)] + self.lr * (reward + self.gamma *td_error)
    
    def learn(self):
        states, actions, rewards, next_states, done =  self.memory.sample(self.batch_size)
        # update Q function
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, done):
            max_next_state = np.max(self.Q[next_state])
            td_error = self.Q[state, action] - max_next_state
            self.Q[(state,action)] = self.Q[(state,action)] + self.lr * (reward + self.gamma*  td_error)
    
    def compute_reward_loss(self, episode=10):
        """
        use the env to create the real reward and compare it to the predicted
        reward of the model
 
        """
        self.env.seed(np.random.randint(0,10))
        reward_loss = 0
        reward_list = []
        for epi in range(episode):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.trained_Q[state])
                next_state, reward, done, _ = self.env.step(action)
                predict_reward = self.r[state, action]
                reward_list.append((reward, predict_reward))
                if done: 
                    break
        reward_loss =([abs(r[0] - r[1]) for r in reward_list]  )
        reward_loss_length = len(reward_loss)
        reward_loss = sum(reward_loss) / reward_loss_length
        self.last_100_reward_errors.append(reward_loss)
        average_loss = np.mean(self.last_100_reward_errors)
        print("average mean loss ", average_loss)
        self.writer.add_scalar('Reward_loss', reward_loss, self.steps)
        self.writer.add_scalar('Average_Reward_loss', average_loss, self.steps)
        #print(reward_loss)

    
    def invers_q(self, continue_train=False):
        self.memory.load_memory("memory") 
        self.load_q_table()
        if not continue_train:
            print("clean policy")
            self.Q = np.zeros([self.state_size, self.action_size])
        mkdir("", "inverse_policy") 
        for epi in range(1, self.episodes_qinverse + 1):
            self.steps += 1
            text = "Inverse Episode {} \r".format(epi)
            # print(text, end = '')
            if epi % self.eval_q_inverse == 0:
                self.start_reward()
                self.memory.save_memory("inverse_policy")
                self.save_q_table("inverse_Q")
                self.save_r_table()
                self.render_env = False
                self.eval_policy(use_inverse=True, episode=5)
                self.eval_policy(use_expert=True, episode=5)
                self.render_env =False
            state, action, r, next_state, _ = self.memory.sample(1)
            action = action[0][0]
            state = state[0][0]
            next_state = next_state[0][0]
            self.counter[state, action] += 1
            total_num = np.sum(self.counter[state,:])
            action_prob = self.counter[state] / total_num
            assert(np.isclose(np.sum(action_prob),1))
            # update Q shift 
            Q_shift_target = self.lr_sh * (self.gamma_iql * np.max(self.Q_inverse[next_state]))
            #print("q values", self.Q[state])
            self.Q_shift[state, action] = ((1 - self.lr_sh) * self.Q_shift[state, action]) + Q_shift_target
            # compute n a
            if action_prob[action] == 0:
                action_prob[action] =  np.finfo(float).eps
            n_a = np.log(action_prob[action]) - self.Q_shift[state, action]
            
            # update reward function
            self.update_r(state, action, n_a, action_prob)
            #self.debug_train()
            # update Q function
            self.update_q(state, action, next_state)
            # self.policy_diff(state, action)

    def update_q(self, state, action, next_state):
        q_old = (1 - self.lr_iql_q) * self.Q_inverse[state, action]
        q_new = self.lr_iql_q *(self.r[state, action] + (self.gamma_iql * np.max(self.Q_inverse[next_state])))
        #print("q old ", q_old)
        #print("q_new", q_new)
        #print("q invers ", q_old + q_new)
        self.Q_inverse[state, action] = q_old + q_new
        
    def update_r(self, state, action, n_a, action_prob):
        r_old = (1 - self.lr_iql_r) * self.r[state, action]
        part1 = n_a
        #print("part1", n_a)
        part2 = self.ratio * self.sum_over_action(state, action, action_prob)
        r_new = self.lr_iql_r * (part1 + part2)
        #print("r old ", r_old)
        #print("r_new", r_new)
        self.r[state, action] = r_old + r_new       
    
    def sum_over_action(self, state, a, action_prob):
        res = 0
        for b in range(self.action_size):
            if b == a:
                continue
            res = res + (self.r[state, b] - self.compute_n_a(state, b, action_prob))
        return res

    def compute_n_a(self, state, action, action_prob):
        if action_prob[action] == 0:
            action_prob[action] = np.finfo(float).eps
        return np.log(action_prob[action]) - self.Q_shift[state, action]


    def start_reward(self):
        self.env.seed = 1
        
        state = self.env.reset()
        print(state)
        ns, r, d, _ = self.env.step(0)
        np.set_printoptions(precision=2)
        print(" expert q {}".format(self.trained_Q[state])) 
        print("inverse q {}".format(self.Q_inverse[state]))
        return 

    
    def eval_policy(self, random_agent=False, use_expert=False, use_debug=False, use_inverse=False,episode=10):
        if use_expert:
            self.load_q_table()
        total_steps = 0
        total_reward = 0
        total_penetlies = 0
        for i_episode in range(1, episode + 1):
            score = 0
            steps = 0
            state = self.env.reset()
            done  = False
            penelty = 0
            while not done:
                steps += 1
                if use_expert:
                    action = np.argmax(self.trained_Q[state])
                elif random_agent:
                    action = self.env.action_space.sample() 
                elif use_debug:
                    action = np.argmax(self.debug_Q[state])
                elif use_inverse:
                    action = np.argmax(self.Q_inverse[state])
                else:
                    action = self.act(state, 0, True)
                
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if self.render_env:
                    self.env.render()
                    time.sleep(0.1)
                score += reward
                if reward == -10:
                    penelty += 1
                if done:
                    total_steps += steps
                    total_reward += score
                    total_penetlies += penelty
                    break
        if self.render_env:
            self.env.close()
        aver_steps = total_steps / episode
        average_reward = total_reward / episode
        aver_penelties = total_penetlies / episode
        
        if use_expert:
            print("Expert avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))

        elif random_agent:
            print("Random Eval avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))
        
        elif use_inverse:
            print("Inverse q Eval avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))
        
        else:    
            print("Eval avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))
            self.writer.add_scalar('Eval_Average_steps', aver_steps, self.steps)
            self.writer.add_scalar('Eval_Average_reward', average_reward, self.steps)
            self.writer.add_scalar('Eval_Average_penelties', aver_penelties, self.steps)
       
    def save_q_table(self, table="Q", filename="policy"):
        mkdir("", filename)
        if table == "Q":
            with open(filename + '/Q.npy', 'wb') as f:
                np.save(f, self.Q)
        if table =="inverse_Q":
            with open(filename + '/Inverse_Q.npy', 'wb') as f:
                np.save(f, self.Q_inverse)

    def load_q_table(self, table="Q", filename="policy"):
        if table == "Q":
            with open(filename + '/Q.npy', 'rb') as f:
                self.Q = np.load(f)
        if table == "inverse_Q":
            with open(filename + '/Inverse_Q.npy', 'rb') as f:
                self.Q_inverse = np.load(f)

        self.trained_Q = self.Q
    
    def save_r_table(self, filename="reward_function"):
        mkdir("", filename)
        with open(filename + '/r.npy', 'wb') as f:
            np.save(f, self.r)

    def load_r_table(self, filename="reward_function"):
        with open(filename + '/r.npy', 'rb') as f:
            self.r = np.load(f)


    def eval_inverse(self):
        self.load_q_table(table= "inverse_Q")
        for i_episode in range(1, 11):
            score = 0
            steps = 0
            penelties = 0
            state = self.env.reset()
            done  = False
            while not done:
                steps += 1
                print(self.Q_inverse)
                action = np.argmax(self.Q_inverse[state])
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                if reward == -10:
                    penelties += 1
                state = next_state
            print("Inverse  steps {} reward  {:.2f}  penelty {} ".format(steps, score, penelties))




    def policy_diff(self, state, expert_action):

        self.trained_Q = self.Q

    def create_expert_policy(self):
        self.load_q_table()
        self.trained_Q = self.Q
        for i_episode in range(1, self.expert_buffer_size + 1):
            text = "create Buffer {} of {}\r".format(i_episode, self.expert_buffer_size)
            print(text, end=" ")
            state = self.env.reset()
            if state == 184:
                print("yes ")
            done  = False
            score = 0
            while True:
                action = self.act(state, 0, True)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.memory.add(state, action, reward, next_state, done, done)
                state = next_state
                if done:
                    #print("reward ", score)
                    break
        self.memory.save_memory("memory")


    def policy_diff(self, state, expert_action):
        action = np.argmax(self.Q_inverse[state])
        if action == expert_action:
            print("Episode {} Reward {:.2f} Average Reward {:.2f} steps {}  epsilon {:.2f}".format(i_episode, score, average_reward, steps, self.epsilon))
            self.writer.add_scalar('Average_reward', average_reward, self.steps)
            self.writer.add_scalar('Train_reward', score, self.steps)
        self.trained_Q = self.Q
        self.memory.save_memory("memory")
        
        
    def debug_train(self):
        """

        use the trained reward function to train the agent

        """
        state = self.env.reset()
        done  = False
        score = 0
        self.steps += 1
        epsiode_steps =  0
        while True:
            action = self.act(state, 0, True)
            next_state, _, done, _ = self.env.step(action)
            reward = self.r[state, action]
            self.optimize(state, action, reward, next_state, debug=True)

            score += reward
            epsiode_steps += 1
            if done:
                break
            state = next_state

        self.total_reward += score
        average_reward = self.total_reward / self.steps
        print("Episode {} Reward {:.2f} Average Reward {:.2f}  epi steps {}".format(self.steps, score, average_reward, epsiode_steps))


    def train(self):
      
        total_timestep = 0
        for i_episode in range(1, self.episode + 1):
            score = 0
            state = self.env.reset()
            done  = False
            steps = 0
            while not done:
                self.steps +=1
                steps += 1
                total_timestep += 1
                action = self.act(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.optimize(state, action, reward, next_state)
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay * i_episode)
                
                if done:
                    break
                state = next_state
            
            if i_episode % self.eval_frq == 0:
                self.eval_policy()
            
            self.total_reward += score
            average_reward = self.total_reward / i_episode
            print("Episode {} Reward {:.2f} Average Reward {:.2f} steps {}  epsilon {:.2f}".format(i_episode, score, average_reward, steps, self.epsilon))
            self.writer.add_scalar('Average_reward', average_reward, self.steps)
            self.writer.add_scalar('Train_reward', score, self.steps)
        self.trained_Q = self.Q
        
