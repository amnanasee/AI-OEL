from random import choices, randint, randrange, random
from typing import List, Tuple, Callable, NamedTuple

Genome = List[int]
Population = List[Genome]

Thing = NamedTuple('Thing', [('name', str), ('value', int), ('weight', int)])

def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be the same length")
    
    weight = 0
    value = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            if weight > weight_limit:
                return 0
    return value

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

def population_fitness(population: Population, fitness_func: Callable[[Genome], int]) -> int:
    return sum([fitness_func(genome) for genome in population])

def selection_pair(population: Population, fitness_func: Callable[[Genome], int]) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def run_evolution(
        populate_func: Callable[[], Population],
        fitness_func: Callable[[Genome], int],
        fitness_limit: int,
        selection_func: Callable[[Population, Callable[[Genome], int]], Tuple[Genome, Genome]] = selection_pair,
        crossover_func: Callable[[Genome, Genome], Tuple[Genome, Genome]] = single_point_crossover,
        mutation_func: Callable[[Genome], Genome] = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population, 
            key=lambda genome: fitness_func(genome), 
            reverse=True
        )

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
    
    population = sorted(
            population, 
            key=lambda genome: fitness_func(genome), 
            reverse=True
        )

    return population, i
