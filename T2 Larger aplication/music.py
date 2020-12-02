import click
from numpy import random as random
import numpy as np
import math 
from datetime import datetime
from typing import List, Dict
from midiutil import MIDIFile
from pyo import *
import matplotlib.pyplot as plt


#generate_random_population
#genera una poblacion de melodias
#cada melodia tiene las notas que la componen
#cuanto es el tiempo que dura cada nota
#y la fuerza con la que esta es reproducida
def generate_random_population(n_of_individual,numeronotas):
    
    population = []

    for i in range(0,n_of_individual):
        notas=[]
        velocity=[]
        beat=[]
        for i in range(0,numeronotas):
            notas.append(random.randint(0,100))
            beat.append(random.randint(0,5))
            velocity.append(random.randint(0,127))
        melody = {"notes": [notas],"velocity": velocity,"beat": beat }
        population.append(melody)
    return population


#melody_to_events
#para que se puedan reproducir las melodias desde la consola,es necesario
#hacer que estas sean un evento

def melody_to_events(melody) -> [Events]:
    bpm=128
    return [
        Events(
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.001,
            decay=0.05,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]

#con un input como goal,y una melodia que viene de la poblacion creada
#se compara nota, a nota, (con todas las caracteristicas que tiene una nota
#i.e nota, figura musical, fuerza)
#esta comparación se hace mediante una resta, elemento a elemento
#una vez obtenido estas diferencias, se promedian y se multiplican por un escalar
#este escalar se intenta mejorar empiricamente.

def fitness(melody, goal) -> int:
    distancia_notas =  []
    distancia_volumen = []
    distancia_figura_ritmica=[]



    notasGoal=goal['notes'][0]
    notasGenoma = melody['notes'][0]

    volumenGoal = goal['velocity']
    volumenGenoma = melody['velocity']

    figurasGoal = goal['beat']
    figurasGenoma = melody['beat']

    
    for nota in range(0,len(melody['notes'][0])):
        #se buscará minimizar el promedio de los siguientes arreglos
                
        distancia_notas.append(abs(notasGoal[nota] - notasGenoma[nota]))
        distancia_volumen.append(abs(volumenGoal[nota] - volumenGenoma[nota]))
        distancia_figura_ritmica.append(abs(figurasGoal[nota] - figurasGenoma[nota]))
            
    mean_notas = sum(distancia_notas)
    mean_volumen = sum(distancia_volumen)/len(melody['notes'][0])
    mean_figura_ritmica = sum(distancia_figura_ritmica)
    rating = []
    rating.append(mean_notas*0.45)
    rating.append(mean_volumen*0.1)
    rating.append(mean_figura_ritmica)
    return sum(rating)


#tournament_selection
#metodo para seleccionar los individuos que seran usados para el cross-over
#

def tournament_selection(population, n_of_individual,goal):
    rand_individuals=random.sample(population,n_of_individual)
    individuals_fit=[]
    for i in rand_individuals:
        fit=fitness(i,goal)
        individuals_fit.append((i,fit))
    
    sorted_population_fitness = sorted(individuals_fit, key=lambda e: e[1], reverse=True)
    best=[sorted_population_fitness[0][0], sorted_population_fitness[1][0]]
    return best

#crossover
#metodo que crea un individuo nuevo, a partir de dos padres con un fitness mejor que los demas
def crossover(individual1, individual2):
    n_notas = len(individual1['notes'])
    slice1=random.randint(0,n_notas)
    
    Nuevo_indi_1= {
        "notes": individual1['notes'][0:slice1]+individual2['notes'][slice1:],
        "velocity": individual1['velocity'][0:slice1]+individual2['velocity'][slice1:],
        "beat": individual1['beat'][0:slice1]+individual2['beat'][slice1:]
    }
    
    return Nuevo_indi_1

#mutation
#metodo que cambia una nota(con todos sus atributos), dependiendo de su probabilidad

def mutation(individual,mutation_rate):
    space=np.random.choice([1, 0], size = 100, p=[mutation_rate,1-mutation_rate])
    prob=np.random.choice(space,1)
    gen_a_cambiar=random.randint(0,len(individual['notes'][0])-1)
    if(prob[0]==1):
        individual['notes'][0][gen_a_cambiar]=random.randint(0,100)
        individual['beat'][gen_a_cambiar]=random.randint(0,5)
        individual['velocity'][gen_a_cambiar] = random.randint(0,127)
    return individual

#save_melody_to_midi
#si el resultado es satisfactorio, es posible guardar este
#como archivo mid

def save_melody_to_midi(melody,filename: str):
    
    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
       raise ValueError

    mf = MIDIFile(1)

    track = 0
    channel = 0

    time = 0.0
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)

    for i, vel in enumerate(melody["velocity"]):
        if vel > 0:
            for step in melody["notes"]:
                mf.addNote(track, channel, step[i], time, melody["beat"][i], vel)

        time += melody["beat"][i]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        mf.writeFile(f)





#goal1,2 y 3, son melodias, escritas a partir de canciones de diversos origenes
#goal 1 viene del main-theme del videojuego Child of light, desarrollado por Ubisoft Montreal
#goal 2, fue un intento de adaptar una de las melodias de la pelicula Kimi no Nawa
#goal 3, es otro intento de adaptar el main theme del videojuego The legend of Zelda

#todas estas melodias fueron transcritas manualmente desde partituras dispobibles desde la web musecore.com


goal1 = {'notes': [[71,69, 67,66, 67, 69, 67, 66, 64, 64, 72, 71, 66, 67, 69,67,66,64,64,64]],
         'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,127, 127, 127, 127, 127, 127, 127, 127],
         'beat': [2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 6.0, 1.0, 3.0, 1.0, 2.0, 1.0,1.0,0.5,0.5,2.0,0.5,1.0]}
goal2 = {'notes': [[67,67, 67,67, 67, 64, 62, 62, 60, 64, 72, 60, 67, 67, 60,72,71,69,67,67]],
         'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 0, 127,127, 127, 127, 127, 127, 127, 127, 127, 127],
         'beat': [2.0, 1.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 6.0, 1.0, 1.0, 1.5, 0.5, 1.0,0.5,0.5,1.0,0.5,0.5,1.0]}
goal3 = {'notes': [[59,53, 59 , 59, 61, 63, 64, 66, 66, 66, 67, 69, 71, 71, 71,71,71,69,67,67]],
         'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 127,127, 127, 127, 127, 127, 127, 127, 127, 127, 127],
         'beat': [2.0, 3.0, 1.0, 0.5, 0.5, 0.5, 0.5, 4.0, 1.5, 0.5, 1.0, 1.0, 1.0, 4.0,1.0,1.0,1.0,1.0,1.0,1.0]}
goles = [goal1, goal2, goal3]

bpm = 88


#las siguientes comandos sirven para cambiar las variables desde la consola

@click.command()
@click.option("--n-population", default=100, prompt='Population size:', type=int)
@click.option("--mutaciones", default=15, prompt='Number of mutations:', type=int)
@click.option("--mutation-rate", default=0.5, prompt='Mutations probability:', type=float)
@click.option("--n-iteraciones", default=5000, prompt='maximo generations:', type=int)
@click.option("--goal", default=0, prompt='goal:[0-2]', type=int)
@click.option("--reproduction-rate", default=30, prompt='reproductionrate:', type=int)


#main, usa todas las funciones y variables previamente definidas

def main(n_population: int,mutaciones:int, mutation_rate:int, n_iteraciones:int,reproduction_rate: int ,goal:int):

    goal = goles[goal]
    k = len(goal['notes'][0])

    
    population=generate_random_population(n_population,k)   #se crea una poblacion de melodias
    s = Server().boot()

    events = melody_to_events(goal) #la melodia goal, es reproducida
    for e in events:
        e.play()
    s.start()
    input("here is the goal …")
    s.stop()
    for e in events:
        e.stop()
    time.sleep(1)
    resultados=[]
    best0 = []
    iteration = 0 #se inicializa la variable que representa la generación en la que se encuentra el algoritmo
    iteration_chart=[]
    while(iteration<n_iteraciones):
        iteration+=1
        iteration_chart.append(iteration)
        fitness_array=[]
        for i in population: #para cada melodia
            i_fitness=fitness(i,goal) #calculate the fitness
            oo = i
            resultados.append((i_fitness,oo)) #guardamos en un arreglo
            print(i_fitness)
            if(i_fitness<0.5): #solution found
                final = i
                save_melody_to_midi(i,"resultado") #en caso de que sea  solucion deseable, se reproduce en la consola y se guarda como archivo
                best0.append(i)
                events = melody_to_events(final)
                for e in events:
                    e.play()
                s.start()
                input("here is the result …")
                s.stop()
                for e in events:
                    e.stop()
                time.sleep(1)
                break
            fitness_array.append(i_fitness)



        for i in range(0,reproduction_rate): #cambia la tasa de reproduccion           
            n_of_selected=5
            selected=tournament_selection(population,n_of_selected,goal) #tournament selection
            nex_generation=mutation(crossover(selected[0],selected[1]), mutation_rate) #crossover and mutation for the child
            random.shuffle(population)
            replace_index=random.randint(0,len(population)-1)
            population.remove(population[replace_index]) #replace old random individual
            population.append(nex_generation)   #add new generation individual

        
    sorted_population_fitness = sorted(resultados, key=lambda e: e[0], reverse=False)#en caso de que no se encuentre solucion deseada, se usará la que haya tenido menor fitnes (i.e la que este
        #mas cerca del goal)
    events = melody_to_events(sorted_population_fitness[0][1])
    #save_melody_to_midi(sorted_population_fitness[0][1],"resultado2")
    for e in events:#se reproduce la potencial solucion, y se guarda como archivo
        e.play()
    s.start()
    input("here is the result …")
    s.stop()
    for e in events:
        e.stop()
    time.sleep(1)
    return 0

    #plot
    plt.plot(iteration_chart,sorted_population_fitness,label="best fitness")
    plt.xlabel('iteration/generation')
    plt.ylabel('fitness')
    plt.title('fitness vs generation')
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    main()




