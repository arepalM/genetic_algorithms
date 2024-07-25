from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


# Simply supported beam set parameters #
L   = 10000   # Length of beam (m)
P   = 100     # Concentrated force (N)
La  = 5000    # Location of concentrated force from left support
Lb  = L-La    # Location of concentrated force from right support
x   = La      # Location of interest for maximum deflection
rho = 7.85E-6 # Density (kg/mm^3)

PPKG = 10     # $/kg cost of material


# Simply supported beam optimizable parameters #
E_range = [100E3, 300E3]    # Young's Modulus (MPa)
b_range = [5, 20]           # web thickness 
B_range = [10, 50]          # flange width
h_range = [5, 20]           # flange thickness
H_range = [10, 50]          # web height

# Hyperparameters for GA
random_seed = 42
random.seed(random_seed)

POPULATION_SIZE = 100
P_CROSSOVER = 0.7
P_MUTATION = 0.3
MAX_GENERATIONS = 200

###########################################################################################################################
 #                                                  MAIN BODY                               
###########################################################################################################################

toolbox = base.Toolbox()

#creator.create("FitnessMax",base.Fitness,weights=(1.0,)) # create Fitness class for maximization
creator.create("FitnessMulti",base.Fitness,weights=(-1.0,-1.0)) # create Fitness class for minimizing (deflection,cost)
creator.create("Individual",list,fitness=creator.FitnessMulti) 

Emin = E_range[0] 
Emax = E_range[1]
bmin = b_range[0]
bmax = b_range[1]
Bmin = B_range[0]
Bmax = B_range[1] 
hmin = h_range[0]
hmax = h_range[1]
Hmin = H_range[0]
Hmax = H_range[1]

toolbox.register("attr_E", random.randint, Emin, Emax)
toolbox.register("attr_Wt", random.randint, bmin, bmax)
toolbox.register("attr_TFw", random.randint, Bmin, Bmax)
toolbox.register("attr_TFt", random.randint, hmin, hmax)
toolbox.register("attr_Wh", random.randint, Hmin, Hmax)


toolbox.register("individualCreator", tools.initCycle, creator.Individual,
                 (toolbox.attr_E, toolbox.attr_Wt, toolbox.attr_TFw, toolbox.attr_TFt, toolbox.attr_Wh), n=1)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Objective functions


def calc_cost(individual):
    _, Wt, TFw, TFt, Wh = individual

    return PPKG * L * rho * Wt * TFw * TFt * Wh,

def calc_deflection(individual):
    E, Wt, TFw, TFt, Wh = individual
    
    #BFt = TFt
    #BFw = TFw
    Ixx = (TFw*(TFt+TFt+Wh)**3-(TFw-Wt)*(TFt+TFt+Wh-2*TFt)**3)/12

    return ((P*Lb*x)/(6*L*E*Ixx)) * (L**2 - Lb**2 - x**2),

def constraint_cost(individual):
    E, Wt, TFw, TFt, Wh = individual

    cost = PPKG * L * rho * Wt * Wh * 2*(TFw * TFt)
    solid_beam_cost = PPKG * rho * L * bmin * Bmax
    return cost <= solid_beam_cost

def constraint_deflection(individual):
    E, Wt, TFw, TFt, Wh = individual
    #BFt = TFt
    #BFw = TFw
    Ixx = (TFw*(TFt+TFt+Wh)**3-(TFw-Wt)*(TFt+TFt+Wh-2*TFt)**3)/12

    delta = (P*Lb*x)/(6*L*E*Ixx) * (L**2 - Lb**2 - x**2)
    return delta <= 10
 
def evaluate(individual):
    cost_trial = calc_cost(individual)[0]
    delta_trial = calc_deflection(individual)[0]

    penalty = 0
    if not constraint_cost(individual):
        penalty += 100
    if not constraint_deflection(individual):
        penalty += 100
    
    return cost_trial + penalty, delta_trial + penalty

# Register function with toolbox and give it an alias
toolbox.register("evaluate",evaluate)

# Select the best individual among tournsize (OR built-in selector from DEAP library) randomly chosen individuals
#toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("select", tools.selNSGA2)

toolbox.register("mutate",tools.mutUniformInt,low=(E_range[0],b_range[0],B_range[0],h_range[0],H_range[0]),
                up = ([E_range[1],b_range[1],B_range[1],h_range[1],H_range[1]]),indpb=1.0/10)  

# NOTE: tools.mutUniformInt() instead of tools.mutFlipBit() may be good for Abaqus material parameter list
# https://deap.readthedocs.io/en/master/api/tools.html#mutation

def main():
    generationCounter = 0
    # Create initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    fitnessValues = list(map(toolbox.evaluate,population))

    # Since order is preserved between fitnessValues and the individuals in the population (a list), use zip() to combine them and assign each individual its corresponding fitness tuple
    for individual, fitnessValue in zip(population,fitnessValues):
        individual.fitness.values = fitnessValue

    # Since we have a dual objective fitness function (i.e., the sum of the individual's entries), we need to extract whole tuple
    #fitnessValues = [individual.fitness.values[0] for individual in population]
    fitnessValues = [individual.fitness.values for individual in population]

    # Determine the max and mean fitness values of generation
    minFitnessValues_deflection = []
    minFitnessValues_cost = []
    meanFitnessValues_deflection = []
    meanFitnessValues_cost = []

    while generationCounter < MAX_GENERATIONS:
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
        #fitnessValues = [ind.fitness.values[0] for ind in population]
        fitnessValues = [ind.fitness.values for ind in population]

        # Calculate max and mean stats
        minFitness_deflection  = min(fitnessValues[0])
        minFitness_cost        = min(fitnessValues[1])
        meanFitness_deflection = sum(fitnessValues[0]) / len(population)
        meanFitness_cost       = sum(fitnessValues[1]) / len(population)
       
        # Append stat values to max and mean lists to show evolution
        minFitnessValues_deflection.append(minFitness_deflection)
        minFitnessValues_cost.append(minFitness_cost)
        meanFitnessValues_deflection.append(meanFitness_deflection)
        meanFitnessValues_cost.append(meanFitness_cost)

        # Print out stats
        print(" - Generation {}: Min Cost = {}, Average Cost = {}".format(generationCounter,abs(minFitness_cost),abs(meanFitness_cost)))
        print(" - Generation {}: Min Deflection = {}, Average Deflection = {}".format(generationCounter,abs(minFitness_deflection),abs(meanFitness_deflection)))

        # Locate index of the (first) best individual using the max fitness value just found
        all_deflections = list(map(itemgetter(0), fitnessValues))
        all_costs = list(map(itemgetter(1), fitnessValues))
        best_deflection_index = all_deflections.index(min(all_deflections))
        best_cost_index = all_costs.index(min(all_costs))
        print("Lowest Deflection Individual = ", *population[best_deflection_index],"\n")
        print("Lowest Cost Individual = ", *population[best_cost_index],"\n")

        # Use matplotlib to plot some of the stats
        plt.figure(1)
        plt.plot([abs(number) for number in minFitnessValues_cost],color='red')
        #plt.plot(meanFitnessValues_cost,color='green')
        plt.xlabel('Generation')
        plt.ylabel('Avg Cost')
        plt.title('Avg Cost over Generations')
        plt.figure(2)
        plt.plot([abs(number) for number in minFitnessValues_deflection],color='blue')
        #plt.plot(meanFitnessValues_deflection,color='black')
        plt.ylabel('Avg Deflection')
        plt.title('Average Deflection over Generations')

if __name__ == "__main__":
        main()
        plt.show()