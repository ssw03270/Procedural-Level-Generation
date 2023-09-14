import torch

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np
import argparse

from network import PPO_discrete
from train import str2bool

def pth2onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')

    parser.add_argument('--seed', type=int, default=209, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=256, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
    parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
    opt = parser.parse_args()

    kwargs = {
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "lr": opt.lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg":opt.l2_reg,
        "entropy_coef":opt.entropy_coef,  #hard env needs large value
        "adv_normalization":opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    model = PPO_discrete(**kwargs)
    episode = 300000
    model.load(episode=episode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 16000, requires_grad=True).to(device)

    with torch.no_grad():
        model.actor.eval()
        torch.onnx.export(model.actor,
                          x,
                          './model/ppo_{}.onnx'.format(episode),
                          export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                          opset_version=10,  # 모델을 변환할 때 사용할 ONNX 버전
                          do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                          input_names=['input'],  # 모델의 입력값을 가리키는 이름
                          output_names=['output'],  # 모델의 출력값을 가리키는 이름
                          dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                        'output': {0: 'batch_size'}},
                          verbose=True)

def test():
    ENV_PATH = './Unity_PLG/Build/'
    ENV_NAME = 'Unity_PLG.exe'
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(
        width=900, height=450
        , time_scale=0.00001
    )
    env = UnityEnvironment(file_name=ENV_PATH + ENV_NAME, worker_id=0, no_graphics=False,
                           side_channels=[config_channel], seed=1)

    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    behavior_name = list(env.behavior_specs)[0]
    print("The size of frame is: ", len(env.behavior_specs[behavior_name].observation_specs))
    print("No. of Actions: ", len(env.behavior_specs[behavior_name].action_spec))

    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')

    parser.add_argument('--seed', type=int, default=209, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=256, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
    parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
    opt = parser.parse_args()
    print(opt)

    T_horizon = opt.T_horizon
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps
    max_e_steps = 1000

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = {
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "lr": opt.lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg":opt.l2_reg,
        "entropy_coef":opt.entropy_coef,  #hard env needs large value
        "adv_normalization":opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    model = PPO_discrete(**kwargs)
    episode = 300000
    model.load(episode=episode)

    traj_lenth = 0
    total_steps = 0

    # training loop
    while total_steps < Max_train_steps:
        env.reset()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        s = np.array(decision_steps.obs)
        done = len(terminal_steps.obs[0]) != 0
        steps, ep_r = 0, 0

        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))
            act = [a % 1000, a // 1000]

            env.set_actions(behavior_name, ActionTuple(discrete=np.array([act])))
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            done = len(terminal_steps.obs[0]) != 0

            if not done:
                r = decision_steps.reward[0]
                s_prime = np.array(decision_steps.obs)
            else:
                r = terminal_steps.reward[0]
                s_prime = np.array(terminal_steps.obs)

            s = s_prime
            ep_r += r

            total_steps += 1
        print(ep_r)

    env.close()

if __name__ == '__main__':
    test()