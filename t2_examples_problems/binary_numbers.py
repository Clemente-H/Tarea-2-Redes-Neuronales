import random
import string
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

#function for calculate the fitness of a individal
#individual -> the individual.
#goal -> the goal
def fitness(individual, goal):
    individual_fitness=0
    goal_number=int(goal)
    indx=1
    result=0

    for i in individual[::-1]:
        result+=int(i) * indx
        indx*=2
    individual_fitness=goal_number-result
    return -abs(individual_fitness)

#function for pick two individual with the tournament method
#population -> the array with the entire population.
#n_of_individual -> number of individual to being picked up with the method.
#goal -> the goal
def tournament_selection(population, n_of_individual,goal):
    rand_individuals=random.sample(population,n_of_individual)
    individuals_fit=[]
    for i in rand_individuals:
        fit=fitness(i,goal)
        individuals_fit.append((i,fit))
    
    sorted_population_fitness = sorted(individuals_fit, key=lambda e: e[1], reverse=True)
    best=[sorted_population_fitness[0][0], sorted_population_fitness[1][0]]
    return best


#function for generate a random population of individuals.
#n_of_genes -> the number of genes in the individual.
#n_of_individual -> the number of individual to generate.
def generate_random_population(n_of_genes,n_of_individual):
    population=[]
    for i in range(n_of_individual):
        individual=""
        for j in range(n_of_genes):
            number=random.choice([0,1])
            individual=individual+str(number)

        population.append(individual)
    return population

#function for the crossover of the individuls to generate a new one.
#individual1,2 -> the individual to being crossovered. 
def crossover(individual1, individual2):
    new_individual=""
    slice1=random.randint(0,len(individual1)-1)
    for i in range(0,slice1):
        new_individual=new_individual+individual1[i]
    for i in range(slice1,len(individual1)):
        new_individual=new_individual+individual2[i]
    
    return new_individual

#function for mutate (or not) one gen of an individual
#individual -> the individual
#mutation_rate -> the probability of being mutated
def mutation(individual,mutation_rate):
    space=np.random.choice([1, 0], size = 100, p=[mutation_rate,1-mutation_rate])
    prob=np.random.choice(space,1)
    if(prob[0]==1):
        muted_individual=""
        index_gen=random.randint(0,len(individual)-1)
        random_gen=random.choice(['0','1'])
        for i in range(len(individual)):
            if (i==index_gen):
                muted_individual=muted_individual+random_gen
            else:
                muted_individual=muted_individual+individual[i]
        
        return muted_individual

    return individual

#the genetic algorithm
#n_population -> number of indivuals in population
#goal -> the goal
#n_of_selected -> number of selected with the tournament method
#mutation_rate -> the probability to mutate an individual
#reproduction_rate -> the number of new individuals of each genearation
def genetic_algorithm(n_population, goal, n_of_selected,mutation_rate,reproduction_rate):
    str_goal= 10 #arbitrary /// #math.ceil(math.log2(goal)) -> ideal, but didnt work, it crashes.
    population=generate_random_population(str_goal, n_population)
    goal_selected=0
    fitness_chart_best=[]
    fitness_chart_worst=[]
    fitness_chart_average=[]
    iteration_chart=[]
    iteration=0

    while(goal_selected==0):
        iteration+=1
        iteration_chart.append(iteration)
        fitness_array=[]
        best=-1000000 #arbitrary
        worst=0
        average=0
        for i in population:
            i_fitness=fitness(i,str(goal)) #calculate the fitness
            if(i_fitness==0): #solution found
                goal_selected=i
                fitness_chart_best.append(i_fitness)
                fitness_chart_worst.append(worst)
                break
            if(i_fitness>best):
                best=i_fitness

            if(i_fitness<worst):
                worst=i_fitness

            fitness_array.append(i_fitness)
        average=statistics.mean(fitness_array)
        fitness_chart_average.append(average)
        if(goal_selected!=0):
            break
        fitness_chart_best.append(best)
        fitness_chart_worst.append(worst)
        
        for i in range(0,reproduction_rate):
            selected=tournament_selection(population,n_of_selected,str(goal)) #tournament selection
            nex_generation=mutation(crossover(selected[0],selected[1]), mutation_rate) #crossover and mutation for the child
            random.shuffle(population)
            replace_index=random.randint(0,len(population)-1)
            population.remove(population[replace_index]) #replace old random individual
            population.append(nex_generation)   #add new generation individual

            
    #results
    print("Result: "+str(goal_selected))
    print("Iterations: "+str(len(iteration_chart)))
    print("Best fitness array: " ,fitness_chart_best)
    print("Worst fitness array: ",fitness_chart_worst)

    #plot
    plt.plot(iteration_chart,fitness_chart_best,label="best fitness")
    plt.plot(iteration_chart,fitness_chart_worst, label="worst fitness")
    plt.plot(iteration_chart,fitness_chart_average, label="average fitness")
    plt.xlabel('iteration/generation')
    plt.ylabel('fitness')
    plt.title('fitness vs generation')
    plt.legend()
    plt.show()

    return goal_selected #the goal selected individual (solution)

#test
genetic_algorithm(100,29,5,0.7,10)