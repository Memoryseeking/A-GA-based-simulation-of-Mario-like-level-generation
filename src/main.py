"""
主程序模块
用于测试和运行马里奥关卡生成器
"""

import os
import sys
import time
import numpy as np
import pygame
from src.level import Level
from src.genetic_algorithm import GeneticAlgorithm
from src.visualizer import visualize_level, visualize_evolution, LevelVisualizer, EvolutionVisualizer

def test_level_representation():
    """测试关卡表示"""
    print("Testing level representation...")
    
    # 创建一个随机关卡
    level = Level(width=100, height=15)
    level.random_init()
    
    # 打印关卡信息
    print(f"Level size: {level.width}x{level.height}")
    print(f"Start position: {level.start_pos}")
    print(f"End position: {level.end_pos}")
    
    # 检查关卡有效性
    is_valid = level.is_valid()
    print(f"Level validity: {is_valid}")
    
    # 检查路径存在性
    path_exists = level.check_path_exists()
    print(f"Path existence: {path_exists}")
    
    # 计算统计信息
    stats = level.calculate_statistics()
    print("Level statistics:")
    for key, value in stats.items():
        if key == 'element_counts':
            print("  Element counts:")
            for element, count in value.items():
                print(f"    {element}: {count}")
        else:
            print(f"  {key}: {value}")
    
    # 测试序列化和反序列化
    json_str = level.to_json()
    level2 = Level.from_json(json_str)
    
    # 验证反序列化是否正确
    is_same = np.array_equal(level.grid, level2.grid)
    print(f"Serialization and deserialization test: {'Passed' if is_same else 'Failed'}")
    
    return level

def test_genetic_algorithm(level_width=100, level_height=15, population_size=20, generations=50):
    """测试遗传算法"""
    print("Testing genetic algorithm...")
    
    # 创建遗传算法对象
    ga = GeneticAlgorithm(
        population_size=population_size,
        level_width=level_width,
        level_height=level_height,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=2
    )
    
    # 初始化种群
    print("Initializing population...")
    population = ga.initialize_population()
    print(f"Population size: {len(population)}")
    
    # 评估初始种群
    print("Evaluating initial population...")
    fitness_values = ga.evaluate_population()
    print(f"Initial best fitness: {max(fitness_values):.2f}")
    print(f"Initial average fitness: {sum(fitness_values)/len(fitness_values):.2f}")
    
    # 测试选择操作
    print("Testing selection operation...")
    parents = ga.select_parents(selection_method='tournament')
    print(f"Number of selected parents: {len(parents)}")
    
    # 测试交叉操作
    print("Testing crossover operation...")
    offspring = ga.crossover(parents, crossover_method='single_point')
    print(f"Number of offspring: {len(offspring)}")
    
    # 测试变异操作
    print("Testing mutation operation...")
    mutated = ga.mutate(offspring)
    print(f"Number of mutated offspring: {len(mutated)}")
    
    # 测试精英保留
    print("Testing elitism...")
    new_population = ga.apply_elitism(mutated)
    print(f"New population size: {len(new_population)}")
    
    # 运行完整的进化过程
    print(f"Running evolution process ({generations} generations)...")
    start_time = time.time()
    
    best_level = ga.evolve(
        max_generations=generations,
        selection_method='tournament',
        crossover_method='block',
        mutation_methods=['point', 'block', 'structural'],
        stagnation_limit=10
    )
    
    end_time = time.time()
    print(f"Evolution completed, time taken: {end_time - start_time:.2f} seconds")
    print(f"Final best fitness: {ga.best_fitness:.2f}")
    
    # 打印进化历史
    print("Evolution history:")
    for i, fitness in enumerate(ga.best_fitness_history):
        if i % 5 == 0 or i == len(ga.best_fitness_history) - 1:
            print(f"  Generation {i}: {fitness:.2f}")
    
    return ga, best_level

def test_visualization(level):
    """测试可视化"""
    print("Testing level visualization...")
    
    try:
        # 创建可视化器
        visualizer = LevelVisualizer(width=800, height=600, cell_size=32)
        
        # 绘制关卡
        visualizer.draw_level(level)
        
        # 等待用户关闭窗口
        print("Level visualization window opened, press ESC to close...")
        running = True
        while running:
            running = visualizer.handle_events()
            visualizer.draw_level(level)
            visualizer.clock.tick(60)
        
        visualizer.close()
        print("Level visualization test completed")
        
    except Exception as e:
        print(f"Visualization test error: {e}")

def test_evolution_visualization(ga):
    """测试进化可视化"""
    print("Testing evolution visualization...")
    
    try:
        # 创建进化可视化器
        visualizer = EvolutionVisualizer(width=800, height=600)
        
        # 更新数据
        for i, fitness in enumerate(ga.best_fitness_history):
            avg_fitness = sum(ga.fitness_values) / len(ga.fitness_values) if i == len(ga.best_fitness_history) - 1 else 0
            visualizer.update_data(i, fitness, avg_fitness)
        
        # 绘制可视化
        visualizer.draw()
        
        # 等待用户关闭窗口
        print("Evolution visualization window opened, press ESC to close...")
        running = True
        while running:
            running = visualizer.handle_events()
            visualizer.draw()
            visualizer.clock.tick(30)
        
        visualizer.close()
        print("Evolution visualization test completed")
        
    except Exception as e:
        print(f"Evolution visualization test error: {e}")

def test_diversity(population_size=30, generations=30, samples=5):
    """测试关卡多样性"""
    print("Testing level diversity...")
    
    # 创建多个关卡样本
    levels = []
    
    for i in range(samples):
        print(f"Generating sample {i+1}/{samples}...")
        
        # 创建遗传算法对象
        ga = GeneticAlgorithm(
            population_size=population_size,
            level_width=100,
            level_height=15,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=2
        )
        
        # 运行进化过程
        best_level = ga.evolve(
            max_generations=generations,
            selection_method='tournament',
            crossover_method='block',
            mutation_methods=['point', 'block', 'structural'],
            stagnation_limit=10
        )
        
        levels.append(best_level)
        print(f"Sample {i+1} fitness: {ga.best_fitness:.2f}")
    
    # 计算多样性指标
    diversity_scores = []
    
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            # 计算两个关卡之间的差异
            diff = np.sum(levels[i].grid != levels[j].grid) / (levels[i].width * levels[i].height)
            diversity_scores.append(diff)
    
    avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
    print(f"Average diversity score: {avg_diversity:.4f} (0-1 range, higher means more different)")
    
    return levels

def optimize_parameters():
    """优化算法参数"""
    print("Optimizing algorithm parameters...")
    
    # 测试不同参数组合
    parameter_sets = [
        {"population_size": 50, "crossover_rate": 0.8, "mutation_rate": 0.05},
        {"population_size": 50, "crossover_rate": 0.8, "mutation_rate": 0.2},
        {"population_size": 100, "crossover_rate": 0.8, "mutation_rate": 0.2},
        {"population_size": 50, "crossover_rate": 0.6, "mutation_rate": 0.2}
    ]
    
    results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"Testing parameter set {i+1}/{len(parameter_sets)}: {params}")
        
        # 创建遗传算法对象
        ga = GeneticAlgorithm(
            population_size=params["population_size"],
            level_width=100,
            level_height=15,
            crossover_rate=params["crossover_rate"],
            mutation_rate=params["mutation_rate"],
            elite_size=2
        )
        
        # 运行进化过程
        start_time = time.time()
        best_level = ga.evolve(
            max_generations=30,
            selection_method='tournament',
            crossover_method='block',
            mutation_methods=['point', 'block', 'structural'],
            stagnation_limit=10
        )
        end_time = time.time()
        
        # 记录结果
        results.append({
            "params": params,
            "best_fitness": ga.best_fitness,
            "generations": ga.generation,
            "time": end_time - start_time
        })
        
        print(f"  Best fitness: {ga.best_fitness:.2f}")
        print(f"  Generations: {ga.generation}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
    
    # 找出最佳参数集
    best_result = max(results, key=lambda x: x["best_fitness"])
    print("\nBest parameter set:")
    print(f"  Parameters: {best_result['params']}")
    print(f"  Best fitness: {best_result['best_fitness']:.2f}")
    print(f"  Generations: {best_result['generations']}")
    print(f"  Time taken: {best_result['time']:.2f} seconds")
    
    return best_result["params"]

def run_full_test():
    """运行完整测试"""
    print("=== Mario Level Generator Test ===\n")
    
    # 测试关卡表示
    level = test_level_representation()
    print("\n")
    
    # 测试遗传算法
    ga, best_level = test_genetic_algorithm(generations=30)
    print("\n")
    
    # 测试可视化
    test_visualization(best_level)
    print("\n")
    
    # 测试进化可视化
    test_evolution_visualization(ga)
    print("\n")
    
    # 测试关卡多样性
    diverse_levels = test_diversity(samples=3)
    print("\n")
    
    # 优化参数
    best_params = optimize_parameters()
    print("\n")
    
    print("=== Test Completed ===")
    
    return best_level, diverse_levels, best_params

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            # 运行测试
            run_full_test()
        
        elif command == "generate":
            # 生成单个关卡
            print("Generating level...")
            level = Level(width=100, height=15)
            level.random_init()
            visualize_level(level)
        
        elif command == "evolve":
            # 运行进化过程
            print("Running evolution process...")
            generations = 50
            if len(sys.argv) > 2:
                try:
                    generations = int(sys.argv[2])
                except ValueError:
                    pass
            
            ga = GeneticAlgorithm(
                population_size=30,
                level_width=100,
                level_height=15,
                crossover_rate=0.8,
                mutation_rate=0.2,
                elite_size=2
            )
            
            best_level = ga.evolve(
                max_generations=generations,
                selection_method='tournament',
                crossover_method='block',
                mutation_methods=['point', 'block', 'structural'],
                stagnation_limit=10
            )
            
            # 可视化最佳关卡
            visualize_level(best_level)
            
            # 可视化进化过程
            visualize_evolution(ga)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, generate, evolve [generations]")
    
    else:
        # 默认行为：生成并显示一个关卡
        print("Generating and displaying level...")
        level = Level(width=100, height=15)
        level.random_init()
        visualize_level(level)

if __name__ == "__main__":
    main()
