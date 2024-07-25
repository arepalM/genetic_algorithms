from deap import base
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt

random_seed = 42
random.seed(random_seed)

ONE_MAX_LENGTH = 100
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1

MAX_GENERATIONS = 50

toolbox = base.Toolbox()
toolbox.register("zeroOrOne",random.randint,0,1) # picks random integer between 0 and 1

creator.create("FitnessMax",base.Fitness,weights=(1.0,)) # create Fitness class
creator.create("Individual",list,fitness=creator.FitnessMax) # creates an individual using a list, in which the lsit is the individual's chromosome (e.g., material parameters)

toolbox.register("individualCreator",tools.initRepeat,creator.Individual,toolbox.zeroOrOne,ONE_MAX_LENGTH) # uses tools.initRepeat to create an individual
toolbox.register("populationCreator",tools.initRepeat,list,toolbox.individualCreator) # uses tools.initRepeat to create a population of individuals

# Function to evaluate fitness
def oneMaxFitness(individual):
    return sum(individual), # return a tuple becaue fitness values in DEAP are represented as tuples

# Register function with toolbox and give it an alias
toolbox.register("evaluate",oneMaxFitness)

# Select the best individual among tournsize randomly chosen individuals
toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("mate",tools.cxOnePoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=1.0/ONE_MAX_LENGTH)

# NOTE: tools.mutUniformInt() instead of tools.mutFlipBit() may be good for Abaqus material parameter list
# https://deap.readthedocs.io/en/master/api/tools.html#mutation

def main():
    # Create initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # Calculate the fitness for each individual in the initial population
    # Applies evaluate to each individual in the population, which produces an iterable consisting of the fitness tuple for each individual.
    # Output is a list of tuples
    fitnessValues = list(map(toolbox.evaluate,population))
    # Since order is preserved between fitnessValues and the individuals in the population (a list), use zip() to combine them and assign each individual its corresponding fitness tuple
    for individual, fitnessValue in zip(population,fitnessValues):
        individual.fitness.values = fitnessValue

    # Since we have single objective fitness function (i.e., the sum of the individual's entries), we only need to extract first element of tuple
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # Determine the max and mean fitness values of generation
    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter += 1
        # Select the best individual among tournsize randomly chosen individuals len(population) times
        # Randomly select 3 individuals, choose best, and repeat process n times where n = len(population)
        offspring = toolbox.select(population,len(population))
        #  Cloned the offspring so as to preserve the original generation
        offspring = list(map(toolbox.clone,offspring))
        
        # Use single-point crossover, randomly selecting at which point to crossover chromosome
        # https://medium.com/geekculture/crossover-operators-in-ga-cffa77cdd0c8
        for child1, child2 in zip(offspring[::2],offspring[1::2]):
            if random.random() < P_CROSSOVER:
                # Modify child1 and child2 individuals in place (no need to reassign)
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values
        # Apply mutation - iterating over all offspring items, the mutation operator will be applied at the probability set by the mutation probability constant P_MUTATION.
        # If individual gets mutated, delete fitness value (iff it exists)
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Individuals that were not crossed-over or mutated remain intact, and therefore their existing fitness values which were already calculated in a previous generation
        # do not need to be calculated again. The new individuals (those crossed-over or mutated) will have an empty fitness value and now need to be calculated
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate,freshIndividuals))

        for individual, fitnessValue in zip(freshIndividuals,freshFitnessValues):
            individual.fitness.values = fitnessValue       

        # Now the genetic operators are done, we replace old generation with the new generation
        population[:] = offspring

        # Before moving to next generation, collect fitness values for statistics purposes
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness  = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)

        # Print out stats
        print(" - Generation {}: Max Fitness = {}, Average Fitness = {}".format(generationCounter,maxFitness,meanFitness))

        # Locate index of the (first) best individual using the max fitness value just found
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index],"\n")

        # Use matplotlib to plot some of the stats
        plt.plot(maxFitnessValues,color='red')
        plt.plot(meanFitnessValues,color='green')
        plt.xlabel('Generation')
        plt.ylabel('Max / Average Fitness')
        plt.title('Max and Average Fitness over Generations')

main()
plt.show()