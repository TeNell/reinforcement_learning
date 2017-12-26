"""

Enviroment of Chess_Go

Author. Nell (dongboxiang.nell@gmail.com)
Homepage. https://github.com/TeNell 

"""

import copy
import random
import numpy as np
import matplotlib.pyplot as plt


class Chess_Go:
    def __init__(self, board_size):
        self.board_size = board_size
        #self._build()
        
    def _build(self):
        # draw graph
        self.fig = plt.figure('env_Chess_Go.v2',figsize=(8,8),dpi=80)
        self.ax = self.fig.add_subplot(1,1,1,aspect=1)
        self.ax.set_xlim((0, self.board_size-1))
        self.ax.set_ylim((0, self.board_size-1))

        # set ticks
        ticks_num = np.arange(1, self.board_size+1, 1)
        ticks_x = list(ticks_num.astype(str))
        ticks_y = [chr(i+65) for i in range(self.board_size)]
        plt.xticks(list(ticks_num-1), ticks_x)
        plt.yticks(list(reversed(ticks_num-1)), ticks_y)
        self.ax.grid(True, zorder=0)
        
        self.fig.canvas.draw()
        plt.show(block=False)
    
    def available_direction(self, point_map, point):
        direc = np.zeros(9)
        obsta = []

        for i in range(8):
            x = point[0] + self.direc_list[i][0]
            y = point[1] + self.direc_list[i][1]
            if -1 < x < point_map.shape[0] and -1 < y < point_map.shape[1]:
                if point_map[x,y,0]==0:
                    direc[0]=1; direc[i+1]=1
                else:
                    obsta.append(point_map[x,y,1])        
        
        return direc, obsta
    
    def point_update_random(self, point, direc):
        if direc[0]==0:
            pass
        else:
            chooselist = np.nonzero(direc)[0][1:]
            direc_i = random.choice(chooselist)
            
            x = point[0] + self.direc_list[direc_i-1][0]
            y = point[1] + self.direc_list[direc_i-1][1]

            point[0]=x; point[1]=y

        return point
        
        
    def area_calculator(self, point_map):
        
        count_list = np.zeros(point_map[:,:,1].max()+1, dtype=int)

        for i in range(len(count_list)-1):
            idx = np.where(point_map[:,:,1]==i+1)
            count_list[i+1] = len(idx[0])
    
        idx = np.where(point_map[:,:,1]==0)
        for x,y in zip(idx[0],idx[1]):
            point_queue = []
            point_queue_save = [[],[]]
            point_round = []
            if point_map[x, y, 0]==0:
                point_queue.append([x,y]); point_map[x, y, 0] = 1
                while len(point_queue):
                    
                    x_0, y_0 = point_queue[0]

                    point_queue_save[0].append(x_0); point_queue_save[1].append(y_0)
                    direc, obsta = self.available_direction(point_map, [x_0, y_0])

                    if direc[0]!=0:
                        for i in range(8):
                            if direc[i+1]:
                                point_queue.append([x_0+self.direc_list[i][0], y_0+self.direc_list[i][1]])
                                point_map[x_0+self.direc_list[i][0], y_0+self.direc_list[i][1], 0] = 1

                    point_round += obsta

                    point_queue = point_queue[1:]

                p_set = set(point_round) - {0}
                if len(p_set)==1: 
                    count_list[p_set.pop()] += len(point_queue_save[0])
                else:
                    count_list[0] += len(point_queue_save[0])
        
        
        return count_list
        
        
    def map_update(self, point_map, point):
        point_map[point[0], point[1], 0] = 1
        point_map[point[0], point[1], 1] = point[2]
        return point_map
    
    def reset(self):
        self._build()
        
        self.points_xy=[]
        self.points_xy.append([1,               1,               1])
        self.points_xy.append([1,               self.board_size, 2])
        self.points_xy.append([self.board_size, self.board_size, 3])
        self.points_xy.append([self.board_size, 1,               4])
        
        self.colors = ['blue','yellow','green','black']
        
        self.point_map = np.zeros((self.board_size, self.board_size, 2), dtype=int)
        for p in self.points_xy: p[0] = p[0] - 1; p[1] = p[1] - 1
        
        self.direc_list = [[ 0,  1],
                           [ 1,  1],
                           [ 1,  0],
                           [ 1, -1],
                           [ 0, -1],
                           [-1, -1],
                           [-1,  0],
                           [-1,  1]]
        
        for point in self.points_xy:
            self.point_map = self.map_update(self.point_map, point)
        
        count_list = self.area_calculator(self.point_map.copy())
        
        figure_name = ''
        for num in count_list: figure_name += '    ' + str(num) 
        self.figure_name = figure_name + ' | ' + str(count_list.sum()) + '\n'
        
        observation = [self.point_map]
        return observation
        
         
    def step(self, action):
        points_new = []
        points = copy.deepcopy(self.points_xy)
        
        for point in points:
            direc,_ = self.available_direction(self.point_map, point)
            if point==points[-1] and direc[action]==1:
                x = point[0] + self.direc_list[action-1][0]
                y = point[1] + self.direc_list[action-1][1]
                point[0]=x; point[1]=y
            else:
                point = self.point_update_random(point, direc)            
            self.point_map = self.map_update(self.point_map, point)
            points_new.append(point)
                
        done = points_new==self.points_xy
        self.points_xy = points_new

        #print('done:',done)
        
        count_list = self.area_calculator(self.point_map.copy())
        
        figure_name = ''
        for num in count_list: figure_name += '    ' + str(num) 
        self.figure_name = figure_name + ' | ' + str(count_list.sum()) + '\n'
        
        observation = [self.point_map]
        
        #from IPython import embed;embed()
        # 看 MountainCar 的 reward 规律
        reward = float(count_list[4])
        
        if done: plt.cla()
        
        return observation, reward, done
               
        
    def render(self):
        for i in range(len(self.points_xy)):
            new_point = plt.scatter(self.points_xy[i][0], self.points_xy[i][1], clip_on=False,\
                                    c=self.colors[i], linewidths=5, alpha=1, zorder=3)
            self.ax.add_artist(new_point)      
        self.ax.set_title(self.figure_name)
        self.fig.canvas.draw()
        

if __name__ == "__main__":
    env = Chess_Go(board_size = 21)
    env.reset()
    for i in range(500):
        env.render()
        #action = random.randint(1,8)
        action = 1
        _,_,done=env.step(action)
        if done:break
    print('End.')
    plt.show()




