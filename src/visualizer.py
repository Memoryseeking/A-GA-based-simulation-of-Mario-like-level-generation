"""
Pygame Visualization Module
Used for rendering and visualizing Mario levels
"""

import pygame
import numpy as np
import sys
import os
from src.level import Level, EMPTY, GROUND, BRICK, QUESTION, PIPE_TOP_LEFT, PIPE_TOP_RIGHT
from src.level import PIPE_BODY_LEFT, PIPE_BODY_RIGHT, ENEMY, COIN, START, END

class LevelVisualizer:
    """Level visualization class for rendering Mario levels"""
    
    def __init__(self, width=800, height=600, cell_size=32):
        """
        Initialize visualizer
        
        Parameters:
        width (int): Window width
        height (int): Window height
        cell_size (int): Cell size
        """
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mario Level Generator")
        
        # Color definitions
        self.COLORS = {
            EMPTY: (135, 206, 235),      # Sky blue (air)
            GROUND: (139, 69, 19),       # Brown (ground)
            BRICK: (210, 105, 30),       # Brick red (bricks)
            QUESTION: (255, 215, 0),     # Gold (question blocks)
            PIPE_TOP_LEFT: (0, 128, 0),  # Green (pipes)
            PIPE_TOP_RIGHT: (0, 128, 0),
            PIPE_BODY_LEFT: (0, 100, 0), 
            PIPE_BODY_RIGHT: (0, 100, 0),
            ENEMY: (255, 0, 0),          # Red (enemies)
            COIN: (255, 255, 0),         # Yellow (coins)
            START: (0, 0, 255),          # Blue (start point)
            END: (255, 0, 255)           # Purple (end point)
        }
        
        # Load font
        self.font = pygame.font.SysFont(None, 24)
        
        # View offset (for scrolling)
        self.offset_x = 0
        self.offset_y = 0
        
        # Clock object (to control frame rate)
        self.clock = pygame.time.Clock()
    
    def draw_level(self, level):
        """
        Draw level
        
        Parameters:
            level (Level): Level to draw
        """
        # Fill background
        self.screen.fill(self.COLORS[EMPTY])
        
        # Calculate visible area
        visible_cols = self.width // self.cell_size
        visible_rows = self.height // self.cell_size
        
        start_col = max(0, self.offset_x // self.cell_size)
        end_col = min(level.width, start_col + visible_cols + 1)
        
        start_row = max(0, self.offset_y // self.cell_size)
        end_row = min(level.height, start_row + visible_rows + 1)
        
        # Draw grid and elements
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                x = j * self.cell_size - self.offset_x
                y = i * self.cell_size - self.offset_y
                
                element = level.grid[i, j]
                
                # Draw element
                if element != EMPTY:
                    pygame.draw.rect(self.screen, self.COLORS[element], 
                                    (x, y, self.cell_size, self.cell_size))
                
                # Draw shape of special elements
                if element == QUESTION:
                    # Draw question mark in the center of question blocks
                    text = self.font.render("?", True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                    self.screen.blit(text, text_rect)
                
                elif element == COIN:
                    # Draw coins as circles
                    pygame.draw.circle(self.screen, self.COLORS[COIN], 
                                      (x + self.cell_size//2, y + self.cell_size//2), 
                                      self.cell_size//3)
                
                elif element == ENEMY:
                    # Draw enemies as triangles
                    pygame.draw.polygon(self.screen, self.COLORS[ENEMY], 
                                       [(x + self.cell_size//2, y), 
                                        (x, y + self.cell_size), 
                                        (x + self.cell_size, y + self.cell_size)])
                
                elif element == START:
                    # Draw start point as letter S
                    text = self.font.render("S", True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                    self.screen.blit(text, text_rect)
                
                elif element == END:
                    # Draw end point as flag
                    pygame.draw.rect(self.screen, (200, 200, 200), 
                                    (x + self.cell_size//4, y, self.cell_size//8, self.cell_size))
                    pygame.draw.polygon(self.screen, self.COLORS[END], 
                                       [(x + self.cell_size//4, y), 
                                        (x + self.cell_size, y + self.cell_size//4), 
                                        (x + self.cell_size//4, y + self.cell_size//2)])
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                (x, y, self.cell_size, self.cell_size), 1)
        
        # Draw information
        self.draw_info(level)
        
        # Update display
        pygame.display.flip()
    
    def draw_info(self, level):
        """Draw level information"""
        # Draw current position information
        info_text = f"Offset: ({self.offset_x}, {self.offset_y}) | Press ESC to exit, Arrow keys to scroll"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        # Draw level statistics information
        stats = level.calculate_statistics()
        stats_text = f"Width: {stats['width']} | Height: {stats['height']} | Platforms: {stats['platform_count']} | Gaps: {stats['gap_count']}"
        stats_surface = self.font.render(stats_text, True, (0, 0, 0))
        self.screen.blit(stats_surface, (10, 40))
    
    def handle_events(self):
        """
        Handle events
        
        Returns:
            bool: Whether to continue running
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Handle key presses
        keys = pygame.key.get_pressed()
        scroll_speed = 10
        
        if keys[pygame.K_LEFT]:
            self.offset_x = max(0, self.offset_x - scroll_speed)
        if keys[pygame.K_RIGHT]:
            self.offset_x += scroll_speed
        if keys[pygame.K_UP]:
            self.offset_y = max(0, self.offset_y - scroll_speed)
        if keys[pygame.K_DOWN]:
            self.offset_y += scroll_speed
        
        return True
    
    def run(self, level):
        """
        Run visualizer
        
        Parameters:
            level (Level): Level to visualize
        """
        running = True
        
        while running:
            running = self.handle_events()
            self.draw_level(level)
            self.clock.tick(60)  # Limit frame rate to 60FPS
        
        pygame.quit()
    
    def close(self):
        """Close visualizer"""
        pygame.quit()


class EvolutionVisualizer:
    """Evolution process visualization class for displaying genetic algorithm evolution process"""
    
    def __init__(self, width=800, height=600):
        """
        Initialize evolution visualizer
        
        Parameters:
            width (int): Window width
            height (int): Window height
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Genetic Algorithm Evolution Process")
        
        # Load font
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 32)
        
        # Clock object
        self.clock = pygame.time.Clock()
        
        # Chart area
        self.chart_rect = pygame.Rect(50, 50, width - 100, height - 200)
        
        # History data
        self.generation_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.playability_history = []
        self.diversity_history = []
        self.balance_history = []
        self.aesthetics_history = []
    
    def update_data(self, generation, best_fitness, avg_fitness):
        """
        Update data
        
        Parameters:
            generation (int): Current generation
            best_fitness (float): Best fitness
            avg_fitness (float): Average fitness
        """
        self.generation_history.append(generation)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Update history data for each component
        if isinstance(avg_fitness, dict):
            self.playability_history.append(avg_fitness.get('playability', 0))
            self.diversity_history.append(avg_fitness.get('diversity', 0))
            self.balance_history.append(avg_fitness.get('balance', 0))
            self.aesthetics_history.append(avg_fitness.get('aesthetics', 0))
    
    def draw(self):
        """Draw visualization interface"""
        # Fill background
        self.screen.fill((255, 255, 255))
        
        # Draw title
        title = self.title_font.render("Genetic Algorithm Evolution Process", True, (0, 0, 0))
        title_rect = title.get_rect(center=(self.width // 2, 25))
        self.screen.blit(title, title_rect)
        
        # Draw chart border
        pygame.draw.rect(self.screen, (0, 0, 0), self.chart_rect, 2)
        
        # Draw axis labels
        x_label = self.font.render("Generation", True, (0, 0, 0))
        self.screen.blit(x_label, (self.width // 2, self.height - 50))
        
        y_label = self.font.render("Fitness", True, (0, 0, 0))
        y_label_rect = y_label.get_rect(center=(25, self.chart_rect.centery))
        self.screen.blit(y_label, y_label_rect)
        
        # Draw chart
        if len(self.generation_history) > 1:
            # Calculate scaling factor
            x_scale = self.chart_rect.width / max(1, max(self.generation_history))
            
            # Calculate range of all fitness values
            all_fitness_values = (self.best_fitness_history + self.avg_fitness_history + 
                                self.playability_history + self.diversity_history + 
                                self.balance_history + self.aesthetics_history)
            max_fitness = max(all_fitness_values)
            min_fitness = min(all_fitness_values)
            fitness_range = max(1, max_fitness - min_fitness)
            y_scale = self.chart_rect.height / fitness_range
            
            # Draw best fitness curve
            points = []
            for i, gen in enumerate(self.generation_history):
                x = self.chart_rect.left + gen * x_scale
                y = self.chart_rect.bottom - (self.best_fitness_history[i] - min_fitness) * y_scale
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (255, 0, 0), False, points, 2)
            
            # Draw average fitness curve
            points = []
            for i, gen in enumerate(self.generation_history):
                x = self.chart_rect.left + gen * x_scale
                y = self.chart_rect.bottom - (self.avg_fitness_history[i] - min_fitness) * y_scale
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)
            
            # Draw curves for each component
            colors = {
                'playability': (0, 255, 0),    # Green
                'diversity': (255, 165, 0),    # Orange
                'balance': (128, 0, 128),      # Purple
                'aesthetics': (0, 255, 255)    # Cyan
            }
            
            for history, color in [(self.playability_history, colors['playability']),
                                 (self.diversity_history, colors['diversity']),
                                 (self.balance_history, colors['balance']),
                                 (self.aesthetics_history, colors['aesthetics'])]:
                if len(history) > 0:
                    points = []
                    for i, gen in enumerate(self.generation_history):
                        if i < len(history):
                            x = self.chart_rect.left + gen * x_scale
                            y = self.chart_rect.bottom - (history[i] - min_fitness) * y_scale
                            points.append((x, y))
                    
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, color, False, points, 1)
            
            # Draw legend
            legend_y = self.height - 120
            for name, color in [('Best Fitness', (255, 0, 0)),
                              ('Average Fitness', (0, 0, 255)),
                              ('Playability', colors['playability']),
                              ('Diversity', colors['diversity']),
                              ('Balance', colors['balance']),
                              ('Aesthetics', colors['aesthetics'])]:
                pygame.draw.line(self.screen, color, 
                               (self.width - 150, legend_y),
                               (self.width - 120, legend_y), 2)
                label = self.font.render(name, True, (0, 0, 0))
                self.screen.blit(label, (self.width - 110, legend_y - 5))
                legend_y += 20
            
            # Draw current generation and fitness
            current_gen = self.generation_history[-1]
            current_best = self.best_fitness_history[-1]
            current_avg = self.avg_fitness_history[-1]
            
            gen_text = self.font.render(f"Current Generation: {current_gen}", True, (0, 0, 0))
            self.screen.blit(gen_text, (50, self.height - 80))
            
            best_text = self.font.render(f"Best Fitness: {current_best:.2f}", True, (0, 0, 0))
            self.screen.blit(best_text, (50, self.height - 50))
            
            avg_text = self.font.render(f"Average Fitness: {current_avg:.2f}", True, (0, 0, 0))
            self.screen.blit(avg_text, (250, self.height - 50))
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self):
        """
        Handle events
        
        Returns:
            bool: Whether to continue running
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def run(self):
        """Run visualizer"""
        running = True
        
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(30)  # Limit frame rate to 30FPS
        
        pygame.quit()
    
    def close(self):
        """Close visualizer"""
        pygame.quit()


def visualize_level(level):
    """
    Visualize single level
    
    Parameters:
        level (Level): Level to visualize
    """
    visualizer = LevelVisualizer()
    visualizer.run(level)


def visualize_evolution(ga):
    """
    Visualize evolution process
    
    Parameters:
        ga (GeneticAlgorithm): Genetic algorithm object
    """
    visualizer = EvolutionVisualizer()
    
    for i, fitness in enumerate(ga.best_fitness_history):
        # Calculate average fitness
        if i == len(ga.best_fitness_history) - 1:
            # Extract total value from dictionary
            avg_fitness = sum(f['total'] for f in ga.fitness_values) / len(ga.fitness_values)
        else:
            avg_fitness = 0
        visualizer.update_data(i, fitness, avg_fitness)
    
    visualizer.run()
