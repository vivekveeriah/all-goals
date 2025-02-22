import os, logging, gym
import argparse, sys
from . import logger
from .common.misc_util import set_global_seeds
from .monitor import Monitor
from .a2c import learn
from .common.vec_env.subproc_vec_env import SubprocVecEnv
from .common.atari_wrappers import make_atari, wrap_deepmind
from .policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MasteryPolicy
from .gridworld_env import GridWorld

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, eval_interval, eval_steps, *args, **kwargs):
    '''

    :param env_id: name of the environment, something like Pong-Deterministic-v0
    :param num_timesteps: number of time steps for training the network
    :param seed: set this seed to the random generator
    :param policy: type of policy network for the agent
    :param lrschedule: type of step-size annealing schedule
    :param num_cpu: number of threads or workers to use for training
    :return: nothing
    '''

    # def make_env(rank, is_eval=None):
    #     def _thunk():
    #         env = make_atari(env_id)
    #         env.seed(seed + rank)
    #         env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    #         gym.logger.setLevel(logging.WARN)
    #         return wrap_deepmind(env)
    #     return _thunk

    def make_env(rank):
        def _thunk():
            env = GridWorld()
            return env
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(num_cpu)])

    # policy_fn = None
    # if policy == 'cnn':
    #     policy_fn = CnnPolicy
    # elif policy == 'lstm':
    #     policy_fn = LstmPolicy
    # elif policy == 'lnlstm':
    #     policy_fn = LnLstmPolicy
    # assert policy_fn is not None
    policy_fn = MasteryPolicy
    learn(controller_policy=policy_fn, env=env, seed=seed, eval_env=eval_env, eval_interval=eval_interval, eval_steps=eval_steps, nsteps=1, nstack=1,
          target_update_freq=1000, num_goals_per_update=32, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
          )
    env.close()


def main():
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES ---  use this on mac
    import os
    # set SDL to use the dummy NULL video driver,
    #   so it doesn't need a windowing system.
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='Environment ID', default='PongNoFrameskip-v4')
    # BreakoutNoFrameskip-v4
    parser.add_argument('--env', help='Environment ID', default='fig1_randomwalk_box.mdp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--num-workers', type=int, default=4, help='Number of cpu threads to for distributing the '
                                                                   'learning process')
    # parser.add_argument('--eval', type=bool, action='store_true', help='intermittently evaluate learning?')
    parser.add_argument('--eval-interval', type=int, default=100, help='evaluate model at certain number of updates')
    parser.add_argument('--eval-steps', type=int, default=100, help='evaluate model for these many time steps')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=args.num_workers, eval_interval=args.eval_interval,
          eval_steps=args.eval_steps)


if __name__ == '__main__':
    main()
