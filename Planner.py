import numpy as np
import heapq
import matplotlib.pyplot as plt

# 定义曼哈顿距离启发式
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# A* 路径规划算法
def a_star(start, goal, x_feasible, y_feasible, grid_size):
    rows, cols = grid_size
    directions = [(-1, 0, 'y'), (1, 0, 'y'), (0, -1, 'x'), (0, 1, 'x')]  # 上、下、左、右 + 方向类型
    
    visited = np.zeros((rows, cols), dtype=bool)
    heap = []  # (f = g + h, g, position (i,j), path)
    heapq.heappush(heap, (0 + manhattan(start, goal), 0, start, [start]))
    
    while heap:
        f, g, (i, j), path = heapq.heappop(heap)
        
        if (i, j) == goal:
            return path  # 找到路径
        
        if visited[i, j]:
            continue
        visited[i, j] = True
        
        for di, dj, dir_type in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if dir_type == 'x' and not x_feasible[i, j]:
                    continue
                if dir_type == 'y' and not y_feasible[i, j]:
                    continue
                
                new_g = g + 1
                new_h = manhattan((ni, nj), goal)
                new_path = path + [(ni, nj)]
                heapq.heappush(heap, (new_g + new_h, new_g, (ni, nj), new_path))
    
    return None  # 无路径

# 主函数：生成地图、规划路径并可视化
def main():
    # 定义网格大小（例如 10x10）
    rows, cols = 10, 10
    
    # 创建 x_feasible 地图：True 表示允许 x 轴移动
    x_feasible = np.ones((rows, cols), dtype=bool)
    # 添加一些障碍：不允许 x 移动的区域（例如，垂直墙）
    # x_feasible[3:7, 2] = False  # 列2 从行3到6不允许 x 移动
    # x_feasible[0:7, 5] = False  # 列5 从行0到6不允许 x 移动
    x_feasible[2, 3:7] = False  # 列2 从行3到6不允许 x 移动
    x_feasible[7, 3:7] = False  # 列5 从行0到6不允许 x 移动
    # x_feasible[0, 4:7] = False  # 列2 从行3到6不允许 x 移动

    # 创建 y_feasible 地图：True 表示允许 y 轴移动
    y_feasible = np.ones((rows, cols), dtype=bool)
    # 添加一些障碍：不允许 y 移动的区域（例如，水平墙）
    # y_feasible[2, 3:7] = False  # 行2 从列3到6不允许 y 移动
    # y_feasible[7, 3:7] = False  # 行7 从列3到6不允许 y 移动
    y_feasible[3:7, 2] = False  # 行2 从列3到6不允许 y 移动
    y_feasible[0:7, 5] = False  # 行7 从列3到6不允许 y 移动
    y_feasible[4, 8:10] = False  # 列8 从行4到5不允许 y 移动

    # 起点和终点
    start = (0, 0)  # (row, col)
    goal = (9, 9)
    
    # 规划路径
    path = a_star(start, goal, x_feasible, y_feasible, (rows, cols))
    
    if path is None:
        print("No path found!")
        return
    
    print("Path found:", path)
    
    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    
    # 第一幅图：x_feasible 地图
    axs[0].imshow(~x_feasible, cmap='binary', origin='upper')
    # axs[0].set_title('X Feasible Map (Black: Obstacles)')
    # axs[0].set_xlabel('Columns (x)')
    # axs[0].set_ylabel('Rows (y)')
    axs[0].grid(True)
    
    # 第二幅图：y_feasible 地图
    axs[1].imshow(~y_feasible, cmap='binary', origin='upper')
    # axs[1].set_title('Y Feasible Map (Black: Obstacles)')
    # axs[1].set_xlabel('Columns (x)')
    # axs[1].set_ylabel('Rows (y)')
    axs[1].grid(True)
    
    # 第三幅图：路径规划结果
    # 创建一个主地图显示，叠加障碍（例如，不允许任何移动的点标记为障碍）
    obstacle_map = np.zeros((rows, cols))
    obstacle_map[~x_feasible] = 1  # x 不允许标记
    obstacle_map[~y_feasible] = 1  # y 不允许标记（叠加）
    axs[2].imshow(obstacle_map, cmap='Reds', alpha=0.3, origin='upper')  # 浅红显示障碍
    
    # 绘制路径
    path_array = np.array(path)
    axs[2].plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2)  # 蓝线路径
    axs[2].plot(start[1], start[0], 'go', markersize=10)  # 绿点起点
    axs[2].plot(goal[1], goal[0], 'ro', markersize=10)  # 红点终点
    # axs[2].set_title('Path Planning Result')
    # axs[2].set_xlabel('Columns (x)')
    # axs[2].set_ylabel('Rows (y)')
    axs[2].grid(True)

    # Hide axis ticks/numbers on all subfigures
    for axis in axs:
        axis.set_xticks([])
        axis.set_yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()