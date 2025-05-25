"""
Mario Level Generator Instructions
"""

# Mario level generator based on genetic algorithms

This is a programme to automatically generate Mario style levels using a genetic algorithm.The program is implemented using Python and Pygame and generates levels with playability and variety through evolutionary algorithms.

## Features

- Automatic generation of Mario-style game levels using genetic algorithms
- Level elements include ground, bricks, question mark blocks, pipes, enemies, coins, etc.
- Visual level editing and previewing
- Visualisation of evolutionary processes and adaptation changes
- Multiple parameters can be adjusted to generate levels of different styles and difficulty.

## Python 3.x

- Python 3.x
- Pygame
- NumPy

## Usage

The programme offers several different modes of operation:

### 1. Generate and display a single random level

```bash
python run.py
``

### 2. Run the full test

```bash
python run.py test
``

This will test all components of the level representation, genetic algorithm, visualisation, etc.

### 3. Run the evolution process and display the results

```bash
python run.py evolve [generations]
``

where `[generations]` is an optional parameter specifying the number of generations to evolve (default is 50).

## Description of level elements

- Air/blank space: space the character can pass through
- Ground/Platform: Basic elements that the character can stand on
- Bricks: obstacles that can be destroyed
- Question mark blocks: special blocks that may contain bonuses
- Pipes: can act as obstacles or passages
- Enemies: moving obstacles
- Coins: Basic collectibles
- Starting point: The starting position of the level
- End point: the position at which the level ends

## Genetic algorithm parameters

The genetic algorithm parameters in the program can be adjusted in `src/main.py`:

- `population_size`: population size
- `level_width`: width of the level
- `level_height`: level height
- `crossover_rate`: crossover probability
- `mutation_rate`: mutation probability
- `elite_size`: number of elites

## Instructions

In the level visualisation screen:
- Use the arrow keys to scroll through the level
- Press ESC to exit.

## Project structure

- `run.py`: the main startup script.
- `src/level.py`: level representation module
- `src/genetic_algorithm.py`: genetic algorithm module.
- `src/visualizer.py`: visualisation module
- `src/main.py`: main and test modules

## Extensions and customisations

You can customise level generation by modifying the following sections:

1. the element types and level initialisation logic in `level.py`.
2. the fitness function and genetic operations in `genetic_algorithm.py`.
3. change visualisation styles and interactions in `visualizer.py`

## Notes

- The program uses Pygame for visualisation and requires a graphical interface.
- Audio warnings may appear in some environments, but they do not affect the core functionality.
- The playability of the generated levels depends on the design of the fitness function and the tuning of its parameters.
