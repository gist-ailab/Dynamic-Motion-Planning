
from environments import Environment

from planners import Planner



if __name__=="__main__":
    env = Environment()
    planner = Planner(env)


    episode_num = 10
    for ep in range(episode_num):
        print("Start {} episode".format(ep + 1))
        
        env.reset()
        is_end = False

        while not is_end:






