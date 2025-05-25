"""
Mario Level Representation Module
Contains level data structure, initialization, validity checking, and serialization functions
"""

import numpy as np
import random
import json
import copy
from typing import Dict, List, Tuple, Optional

# Define element type constants
EMPTY = 0          # Air/empty space
GROUND = 1         # Ground/platform
BRICK = 2          # Brick
QUESTION = 3       # Question block
PIPE_TOP_LEFT = 4  # Pipe top left
PIPE_TOP_RIGHT = 5 # Pipe top right
PIPE_BODY_LEFT = 6 # Pipe body left
PIPE_BODY_RIGHT = 7 # Pipe body right
ENEMY = 8          # Enemy
COIN = 9           # Coin
START = 10         # Start point
END = 11           # End flag

# Element name mapping (for debugging and display)
ELEMENT_NAMES = {
    EMPTY: "Empty",
    GROUND: "Ground",
    BRICK: "Brick",
    QUESTION: "Question",
    PIPE_TOP_LEFT: "Pipe Top Left",
    PIPE_TOP_RIGHT: "Pipe Top Right",
    PIPE_BODY_LEFT: "Pipe Body Left",
    PIPE_BODY_RIGHT: "Pipe Body Right",
    ENEMY: "Enemy",
    COIN: "Coin",
    START: "Start",
    END: "End"
}

class Level:
    """Class representing a Mario level"""
    
    def __init__(self, width: int, height: int):
        """
        Initialize a new level
        
        Parameters:
        width (int): Level width
        height (int): Level height
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.start_pos = None
        self.end_pos = None
    
    def initialize_random(self):
        """Initialize the level with random elements"""
        # Clear the grid
        self.grid.fill(EMPTY)
        
        # Add ground
        ground_height = np.random.randint(self.height // 2, self.height - 2)
        self.grid[ground_height:, :] = GROUND
        
        # Add platforms
        num_platforms = np.random.randint(3, 8)
        for _ in range(num_platforms):
            platform_length = np.random.randint(3, 8)
            platform_height = np.random.randint(2, ground_height - 2)
            platform_x = np.random.randint(0, self.width - platform_length)
            self.grid[platform_height, platform_x:platform_x + platform_length] = GROUND
        
        # Add bricks
        num_bricks = np.random.randint(5, 15)
        for _ in range(num_bricks):
            brick_x = np.random.randint(0, self.width)
            brick_y = np.random.randint(0, ground_height - 2)
            if self.grid[brick_y, brick_x] == EMPTY:
                self.grid[brick_y, brick_x] = BRICK
        
        # Add question blocks
        num_questions = np.random.randint(3, 8)
        for _ in range(num_questions):
            question_x = np.random.randint(0, self.width)
            question_y = np.random.randint(0, ground_height - 2)
            if self.grid[question_y, question_x] == EMPTY:
                self.grid[question_y, question_x] = QUESTION
        
        # Add pipes
        num_pipes = np.random.randint(2, 5)
        for _ in range(num_pipes):
            pipe_x = np.random.randint(0, self.width - 2)
            pipe_height = np.random.randint(2, 4)
            pipe_y = ground_height - pipe_height
            self.grid[pipe_y:ground_height, pipe_x] = PIPE_BODY_LEFT
            self.grid[pipe_y:ground_height, pipe_x + 1] = PIPE_BODY_RIGHT
            self.grid[pipe_y, pipe_x] = PIPE_TOP_LEFT
            self.grid[pipe_y, pipe_x + 1] = PIPE_TOP_RIGHT
        
        # Add enemies
        num_enemies = np.random.randint(3, 8)
        for _ in range(num_enemies):
            enemy_x = np.random.randint(0, self.width)
            enemy_y = ground_height - 1
            if self.grid[enemy_y, enemy_x] == GROUND:
                self.grid[enemy_y - 1, enemy_x] = ENEMY
        
        # Add coins
        num_coins = np.random.randint(5, 15)
        for _ in range(num_coins):
            coin_x = np.random.randint(0, self.width)
            coin_y = np.random.randint(0, ground_height - 2)
            if self.grid[coin_y, coin_x] == EMPTY:
                self.grid[coin_y, coin_x] = COIN
        
        # Add start and end points
        start_x = np.random.randint(0, self.width // 4)
        end_x = np.random.randint(3 * self.width // 4, self.width)
        self.grid[ground_height - 1, start_x] = START
        self.grid[ground_height - 1, end_x] = END
        self.start_pos = (start_x, ground_height - 1)
        self.end_pos = (end_x, ground_height - 1)
    
    def is_valid(self) -> bool:
        """
        Check if the level is valid
        
        Returns:
        bool: True if the level is valid, False otherwise
        """
        # Check if start and end points exist
        if self.start_pos is None or self.end_pos is None:
            return False
        
        # Check if there is a valid path from start to end
        return self.has_valid_path()
    
    def has_valid_path(self) -> bool:
        """
        Check if there is a valid path from start to end
        
        Returns:
        bool: True if there is a valid path, False otherwise
        """
        if self.start_pos is None or self.end_pos is None:
            return False
        
        # Use breadth-first search to find a path
        visited = set()
        queue = [self.start_pos]
        
        while queue:
            current = queue.pop(0)
            if current == self.end_pos:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            x, y = current
            
            # Check adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 0 <= new_y < self.height and
                    self.grid[new_y, new_x] != EMPTY and
                    (new_x, new_y) not in visited):
                    queue.append((new_x, new_y))
        
        return False
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate level statistics
        
        Returns:
        Dict: Dictionary containing level statistics
        """
        stats = {
            'width': self.width,
            'height': self.height,
            'element_counts': {},
            'platform_count': 0,
            'gap_count': 0
        }
        
        # Count elements
        for element in range(12):  # 0 to 11
            count = np.sum(self.grid == element)
            stats['element_counts'][ELEMENT_NAMES[element]] = count
        
        # Count platforms
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == GROUND:
                    if x == 0 or self.grid[y, x - 1] != GROUND:
                        stats['platform_count'] += 1
        
        # Count gaps
        for y in range(self.height):
            for x in range(self.width - 1):
                if (self.grid[y, x] == GROUND and
                    self.grid[y, x + 1] != GROUND and
                    x + 1 < self.width):
                    stats['gap_count'] += 1
        
        return stats
    
    def serialize(self) -> str:
        """
        Serialize the level to a string
        
        Returns:
        str: Serialized level
        """
        data = {
            'width': self.width,
            'height': self.height,
            'grid': self.grid.tolist(),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos
        }
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Level':
        """
        Deserialize a level from a string
        
        Parameters:
        data (str): Serialized level
        
        Returns:
        Level: Deserialized level
        """
        data = json.loads(data)
        level = cls(data['width'], data['height'])
        level.grid = np.array(data['grid'])
        level.start_pos = tuple(data['start_pos'])
        level.end_pos = tuple(data['end_pos'])
        return level
    
    def get_section(self, start_col, end_col):
        """
        获取关卡的一个区域切片
        
        参数:
            start_col (int): 起始列
            end_col (int): 结束列
        
        返回:
            numpy.ndarray: 区域切片
        """
        # 确保索引在有效范围内
        start_col = max(0, min(start_col, self.width - 1))
        end_col = max(0, min(end_col, self.width))
        
        # 返回区域切片
        return self.grid[:, start_col:end_col].copy()
    
    def set_section(self, start_col, section):
        """
        设置关卡的一个区域切片
        
        参数:
            start_col (int): 起始列
            section (numpy.ndarray): 要设置的区域切片
        """
        # 确保索引在有效范围内
        start_col = max(0, min(start_col, self.width - 1))
        
        # 计算结束列（不超过关卡宽度）
        end_col = min(start_col + section.shape[1], self.width)
        
        # 计算实际可以复制的列数
        cols_to_copy = end_col - start_col
        
        # 设置区域
        if cols_to_copy > 0:
            self.grid[:, start_col:end_col] = section[:, :cols_to_copy]
            
        # 更新起点和终点位置
        self.is_valid()  # 这会重新检测并更新起点和终点位置
    
    def check_path_exists(self):
        """
        检查从起点到终点是否存在可行路径
        
        返回:
            bool: 是否存在可行路径
        """
        if not self.is_valid():
            return False
        
        # 使用简化的A*算法检查路径
        start = self.start_pos
        end = self.end_pos
        
        # 定义可行走的元素
        walkable = [EMPTY, COIN, END]
        
        # 定义移动方向（上、右、下、左、右上、右下）
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1)]
        
        # 初始化开放列表和关闭列表
        open_list = [start]
        closed_list = []
        
        while open_list:
            current = open_list.pop(0)
            closed_list.append(current)
            
            # 到达终点
            if current == end:
                return True
            
            # 检查相邻位置
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # 检查边界
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    continue
                
                # 检查是否可行走
                if self.grid[nx, ny] not in walkable:
                    continue
                
                # 检查是否已经访问过
                if (nx, ny) in closed_list or (nx, ny) in open_list:
                    continue
                
                # 添加到开放列表
                open_list.append((nx, ny))
        
        return False
    
    def copy(self):
        """
        创建关卡的深拷贝
        
        返回:
            Level: 关卡的深拷贝
        """
        new_level = Level(width=self.width, height=self.height)
        new_level.grid = self.grid.copy()
        new_level.start_pos = self.start_pos
        new_level.end_pos = self.end_pos
        
        return new_level
        
    def clone(self):
        """
        创建关卡的克隆（与copy方法功能相同，为兼容性提供）
        
        返回:
            Level: 关卡的克隆
        """
        return self.copy()
