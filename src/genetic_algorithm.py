"""
Genetic Algorithm Module
Contains population initialization, fitness calculation, selection, crossover, mutation, and elitism functions
"""

import numpy as np
import random
import copy
from src.level import Level, EMPTY, GROUND, BRICK, QUESTION, PIPE_TOP_LEFT, PIPE_TOP_RIGHT
from src.level import PIPE_BODY_LEFT, PIPE_BODY_RIGHT, ENEMY, COIN, START, END
from typing import List, Tuple, Dict, Optional

class GeneticAlgorithm:
    """Genetic algorithm class for generating Mario levels"""
    
    def __init__(self, population_size: int, level_width: int, level_height: int,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 elite_size: int = 2):
        """
        Initialize genetic algorithm
        
        Parameters:
        population_size (int): Population size
        level_width (int): Level width
        level_height (int): Level height
        crossover_rate (float): Crossover probability
        mutation_rate (float): Mutation probability
        elite_size (int): Number of elites
        """
        self.population_size = population_size
        self.level_width = level_width
        self.level_height = level_height
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_values = []
        self.best_fitness = 0
        self.best_level = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self):
        """
        Initialize population
        
        Returns:
        list: Initialized population
        """
        self.population = []
        
        # Create random levels
        for _ in range(self.population_size):
            level = Level(self.level_width, self.level_height)
            level.initialize_random()
            
            # Ensure level is valid
            while not level.is_valid():
                level = Level(self.level_width, self.level_height)
                level.initialize_random()
            
            self.population.append(level)
        
        return self.population
    
    def evaluate_population(self):
        """
        Evaluate population fitness
        
        Returns:
        list: Fitness values for each individual
        """
        self.fitness_values = []
        
        for level in self.population:
            fitness = self.calculate_fitness(level)
            self.fitness_values.append(fitness)
            
            # Update best fitness and level
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_level = level
        
        return self.fitness_values
    
    def calculate_fitness(self, level: Level) -> float:
        """
        Calculate fitness for a level
        
        Parameters:
        level (Level): Level to evaluate
        
        Returns:
        float: Fitness value
        """
        stats = level.calculate_statistics()
        
        # Calculate playability score
        playability = 0
        if level.has_valid_path():
            playability = 1.0
        
        # Calculate diversity score
        diversity = 0
        element_counts = stats['element_counts']
        total_elements = sum(element_counts.values())
        if total_elements > 0:
            # Calculate Shannon diversity index
            proportions = [count / total_elements for count in element_counts.values()]
            diversity = -sum(p * np.log2(p) for p in proportions if p > 0)
        
        # Calculate balance score
        balance = 0
        if stats['platform_count'] > 0 and stats['gap_count'] > 0:
            platform_gap_ratio = stats['platform_count'] / stats['gap_count']
            balance = 1.0 / (1.0 + abs(platform_gap_ratio - 1.0))
        
        # Calculate aesthetics score
        aesthetics = 0
        if element_counts['Coin'] > 0 and element_counts['Enemy'] > 0:
            coin_enemy_ratio = element_counts['Coin'] / element_counts['Enemy']
            aesthetics = 1.0 / (1.0 + abs(coin_enemy_ratio - 2.0))
        
        # Combine scores
        fitness = {
            'playability': playability,
            'diversity': diversity,
            'balance': balance,
            'aesthetics': aesthetics,
            'total': playability * 0.4 + diversity * 0.3 + balance * 0.2 + aesthetics * 0.1
        }
        
        return fitness
    
    def select_parents(self, selection_method: str = 'tournament') -> List[Level]:
        """
        Select parents for reproduction
        
        Parameters:
        selection_method (str): Selection method ('tournament' or 'roulette')
        
        Returns:
        list: Selected parents
        """
        if selection_method == 'tournament':
            return self.tournament_selection()
        else:
            return self.roulette_selection()
    
    def tournament_selection(self, tournament_size: int = 3) -> List[Level]:
        """
        Tournament selection
        
        Parameters:
        tournament_size (int): Size of tournament
        
        Returns:
        list: Selected parents
        """
        parents = []
        
        for _ in range(self.population_size - self.elite_size):
            # Select tournament participants
            tournament = np.random.choice(
                self.population,
                size=tournament_size,
                replace=False
            )
            
            # Find winner
            winner = max(tournament, key=lambda x: self.calculate_fitness(x))
            parents.append(winner)
        
        return parents
    
    def roulette_selection(self) -> List[Level]:
        """
        Roulette wheel selection
        
        Returns:
        list: Selected parents
        """
        # Calculate selection probabilities
        fitness_values = [self.calculate_fitness(level)['total'] for level in self.population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return np.random.choice(self.population, size=self.population_size - self.elite_size)
        
        probabilities = [f / total_fitness for f in fitness_values]
        
        # Select parents
        parents = np.random.choice(
            self.population,
            size=self.population_size - self.elite_size,
            p=probabilities
        )
        
        return parents.tolist()
    
    def crossover(self, parents: List[Level], crossover_method: str = 'single_point') -> List[Level]:
        """
        Perform crossover operation
        
        Parameters:
        parents (list): Parent levels
        crossover_method (str): Crossover method ('single_point' or 'uniform')
        
        Returns:
        list: Offspring levels
        """
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue
            
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if np.random.random() < self.crossover_rate:
                if crossover_method == 'single_point':
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def single_point_crossover(self, parent1: Level, parent2: Level) -> Tuple[Level, Level]:
        """
        Single point crossover
        
        Parameters:
        parent1 (Level): First parent
        parent2 (Level): Second parent
        
        Returns:
        tuple: Two offspring levels
        """
        # Create new levels
        child1 = Level(self.level_width, self.level_height)
        child2 = Level(self.level_width, self.level_height)
        
        # Select crossover point
        crossover_point = np.random.randint(1, self.level_width - 1)
        
        # Perform crossover
        child1.grid = np.concatenate([
            parent1.grid[:, :crossover_point],
            parent2.grid[:, crossover_point:]
        ], axis=1)
        
        child2.grid = np.concatenate([
            parent2.grid[:, :crossover_point],
            parent1.grid[:, crossover_point:]
        ], axis=1)
        
        # Update start and end positions
        child1.start_pos = parent1.start_pos
        child1.end_pos = parent2.end_pos
        child2.start_pos = parent2.start_pos
        child2.end_pos = parent1.end_pos
        
        return child1, child2
    
    def uniform_crossover(self, parent1: Level, parent2: Level) -> Tuple[Level, Level]:
        """
        Uniform crossover
        
        Parameters:
        parent1 (Level): First parent
        parent2 (Level): Second parent
        
        Returns:
        tuple: Two offspring levels
        """
        # Create new levels
        child1 = Level(self.level_width, self.level_height)
        child2 = Level(self.level_width, self.level_height)
        
        # Create mask for crossover
        mask = np.random.random((self.level_height, self.level_width)) < 0.5
        
        # Perform crossover
        child1.grid = np.where(mask, parent1.grid, parent2.grid)
        child2.grid = np.where(mask, parent2.grid, parent1.grid)
        
        # Update start and end positions
        child1.start_pos = parent1.start_pos
        child1.end_pos = parent2.end_pos
        child2.start_pos = parent2.start_pos
        child2.end_pos = parent1.end_pos
        
        return child1, child2
    
    def mutate(self, offspring: List[Level]) -> List[Level]:
        """
        Perform mutation operation
        
        Parameters:
        offspring (list): Offspring levels
        
        Returns:
        list: Mutated offspring
        """
        for level in offspring:
            if np.random.random() < self.mutation_rate:
                self.mutate_level(level)
        
        return offspring
    
    def mutate_level(self, level: Level):
        """
        Mutate a single level
        
        Parameters:
        level (Level): Level to mutate
        """
        # Select random position
        x = np.random.randint(0, self.level_width)
        y = np.random.randint(0, self.level_height)
        
        # Select random element type
        element_type = np.random.randint(0, 12)
        
        # Apply mutation
        level.grid[y, x] = element_type
        
        # Update start and end positions if necessary
        if element_type == 10:  # START
            level.start_pos = (x, y)
        elif element_type == 11:  # END
            level.end_pos = (x, y)
    
    def apply_elitism(self, offspring: List[Level]) -> List[Level]:
        """
        Apply elitism
        
        Parameters:
        offspring (list): Offspring levels
        
        Returns:
        list: New population
        """
        # Sort population by fitness
        sorted_population = sorted(
            self.population,
            key=lambda x: self.calculate_fitness(x)['total'],
            reverse=True
        )
        
        # Select elites
        elites = sorted_population[:self.elite_size]
        
        # Combine elites and offspring
        new_population = elites + offspring
        
        return new_population
    
    def evolve(self, max_generations: int = 50) -> Level:
        """
        Run evolution process
        
        Parameters:
        max_generations (int): Maximum number of generations
        
        Returns:
        Level: Best level found
        """
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population()
        
        # Evolution loop
        for generation in range(max_generations):
            # Select parents
            parents = self.select_parents()
            
            # Perform crossover
            offspring = self.crossover(parents)
            
            # Perform mutation
            offspring = self.mutate(offspring)
            
            # Apply elitism
            self.population = self.apply_elitism(offspring)
            
            # Evaluate new population
            self.evaluate_population()
            
            # Record history
            self.best_fitness_history.append(self.best_fitness)
            avg_fitness = sum(self.calculate_fitness(level)['total'] for level in self.population) / len(self.population)
            self.avg_fitness_history.append(avg_fitness)
        
        return self.best_level
    
    def get_best_level(self) -> Level:
        """
        Get the best level found
        
        Returns:
        Level: Best level
        """
        return self.best_level
