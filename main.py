
from environments import Dynamic2D

# from planners import Planner







if __name__ == '__main__':
    env = Dynamic2D()

    for i in range(10000):
        env.visualize(i)

