import gym
import sys
import json
import argparse
from agent import Agent
from utils import mkdir


def main(args):
    with open (args.param, "r") as f:
        param = json.load(f)
    print("use the env {} ".format(param["env_name"]))
    print(param)
    param["lr_iql_q"] = args.lr_iql_q
    param["lr_iql_r"] = args.lr_iql_r
    param["lr_q_sh"] = args.lr_iql_r
    param["freq_q"] = args.freq_q
    continue_iql = True
    #continue_iql = False
    param["locexp"] = args.locexp
    env = gym.make(param["env_name"])
    if param["env_name"] == "Taxi-v3":
        state_space = env.observation_space.n
        action_space = env.action_space.n
    else:
        state_space = 10000
        action_space = 1
        
    print("State space ", state_space)
    print("Action space ", action_space)
    agent = Agent(state_space, action_space, param)
    
    if args.mode == "create expert policy":
        print("Create expert policy")
        agent.create_expert_policy()
        agent.memory.save_memory("memory")
    elif args.mode == "train q table":
        print("Create q table")
        agent.train()
        agent.save_q_table()
    elif args.mode == "iql":
        print("inverse rl")
        agent.invers_q()

    elif args.mode == "eval iql":
        print("Eval inverse rl")
        agent.train(use_r=True)
        agent.eval_inverse()
    else:
        print("mode not exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="test", type=str)
    parser.add_argument('--lr_iql_q', default=1e-3, type=float)
    parser.add_argument('--lr_iql_r', default=1e-3, type=float)
    parser.add_argument('--lr_q_sh', default=1e-3, type=float)
    parser.add_argument('--freq_q', default=1, type=int)
    parser.add_argument('--mode', default="train q table", type=str)
    arg = parser.parse_args()
    mkdir("", arg.locexp)
    main(arg)
