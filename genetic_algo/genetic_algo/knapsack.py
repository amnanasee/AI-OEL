from functools import partial
from genetic_algorithm import run_evolution, generate_population, fitness, Thing, selection_pair, single_point_crossover, mutation

# Define the knapsack problem's items
things = [
    Thing('Laptop', 500, 2200),
    Thing('Headphones', 150, 168),
    Thing('Coffee Mug', 60, 358),
    Thing('Notepad', 40, 333),
    Thing('Water Bottle', 30, 192),
    Thing('Mints', 5, 25),
    Thing('Socks', 10, 38),
    Thing('Tissues', 15, 88),
    Thing('Phone', 500, 288),
    Thing('Baseball Cap', 100, 70),
] 

# Define the fitness function
def knapsack_fitness(genome, things, weight_limit):
    return fitness(genome, things, weight_limit)

# Function to convert genome to the selected items
def genome_to_things(genome, things):
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result.append(thing.name)
    return result

# Run the genetic algorithm to solve the knapsack problem
population, generations = run_evolution(
    populate_func=partial(generate_population, size=10, genome_length=len(things)),
    fitness_func=partial(knapsack_fitness, things=things, weight_limit=3000),
    fitness_limit=740,
    generation_limit=100
)

# Output the results
print(f"Number of generations: {generations}")
print(f"Best solution: {genome_to_things(population[0],things)}")
