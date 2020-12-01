import click
from numpy import random
import math 
from datetime import datetime
from typing import List, Dict
from midiutil import MIDIFile
from pyo import *
from goals import *

from algorithms.genetic import generate_genome, Genome, selection_pair, single_point_crossover, mutation

BITS_PER_NOTE = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minorM", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]


def int_from_bits(bits: List[int]) -> int:
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))


def genome_to_melody(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: int, key: str, scale: str, root: int) -> Dict[str, list]:
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]

    note_length = 4 / float(num_notes)

    scl = EventScale(root=key, scale=scale, first=root)

    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }

    for note in notes:
        integer = int_from_bits(note)

        if not pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))

        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]

    steps = []
    for step in range(num_steps):
        steps.append([scl[(note+step*2) % len(scl)] for note in melody["notes"]])

    melody["notes"] = steps
    return melody


def genome_to_events(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: bool, key: str, scale: str, root: int, bpm: int) -> [Events]:
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

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


goal1 = {'notes': [[71,69, 67]],'velocity': [127,127, 127],'beat': [2.0, 3.0, 1.0]}

genoma = {'notes': [[45,60, 21]],'velocity': [127,127, 127],'beat': [2.0, 3.0, 1.0]}





def fitness2(genome: Genome, s: Server, num_bars: int, num_notes: int, num_steps: int,
            pauses: bool, key: str, scale: str, root: int, bpm: int, goal) -> int:
    m = metronome(bpm)
    distancia_notas =  []
    distancia_volumen = []
    distancia_figura_ritmica=[]
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)


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
            
    mean_notas = sum(distancia_notas)/len(melody['notes'][0])
    mean_volumen = sum(distancia_volumen)/len(melody['notes'][0])
    mean_figura_ritmica = sum(distancia_figura_ritmica)/len(melody['notes'][0])
    rating = []
    rating.append(mean_notas)
    rating.append(mean_volumen)
    rating.append(mean_figura_ritmica)
    return rating[0]


def metronome(bpm: int):
    met = Metro(time=1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(met, table=t, dur=.25, mul=1)
    freq = Iter(met, choice=[660, 440, 440, 440])
    return Sine(freq=freq, mul=amp).mix(2).out()


def save_genome_to_midi(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                        pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

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














num_bars = 6
num_notes= 3
num_steps= 1
pauses = True
key = random.choice(KEYS)
scale = random.choice(SCALES)
root = 4
bpm = 128

goal1 = {'notes': [[71,69, 67,66, 67, 69, 67, 66, 64, 64, 72, 71, 66, 67, 69,67,66,64,64]],'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127],'beat': [2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 6.0, 1.0, 3.0, 1.0, 2.0, 1.0,1.0,0.5,0.5,2.0,0.5]}
goal2 = {'notes': [[67,67, 67,67, 67, 64, 62, 62, 60, 64, 72, 60, 67, 67, 60,72,71,69,67]],'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 0, 127, 127, 127, 127, 127, 127, 127, 127, 127],'beat': [2.0, 1.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 6.0, 1.0, 1.0, 1.5, 0.5, 1.0,0.5,0.5,1.0,0.5,0.5]}
goal3 = {'notes': [[59,53, 59 , 59, 61, 63, 64, 66, 66, 66, 67, 69, 71, 71, 71,71,71,69,67]],'velocity': [127,127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127],'beat': [2.0, 3.0, 1.0, 0.5, 0.5, 0.5, 0.5, 4.0, 1.5, 0.5, 1.0, 1.0, 1.0, 4.0,1.0,1.0,1.0,1.0,1.0]}

goles = [goal1, goal2, goal3]
def evento_default(goal)-> [Events]:
    melody = goal

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












@click.command()
#@click.option("--num-bars", default=8, prompt='Number of bars:', type=int)
#@click.option("--num-notes", default=4, prompt='Notes per bar:', type=int)
#@click.option("--num-steps", default=1, prompt='Number of steps:', type=int)
#@click.option("--pauses", default=True, prompt='Introduce Pauses?', type=bool)
#@click.option("--key", default="C", prompt='Key:', type=click.Choice(KEYS, case_sensitive=False))
#@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(SCALES, case_sensitive=False))
#@click.option("--root", default=4, prompt='Scale Root:', type=int)
@click.option("--goal", default=0, prompt='goal:', type=int)
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--num-mutations", default=2, prompt='Number of mutations:', type=int)
@click.option("--mutation-probability", default=0.5, prompt='Mutations probability:', type=float)

#@click.option("--bpm", default=128, type=int)










def main(population_size: int, num_mutations: int, mutation_probability: float,goal:int):
    goalF=goles[goal]

    folder = str(int(datetime.now().timestamp()))

    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]

    s = Server().boot()

    population_id = 0

    running = True

    
    events = evento_default(goalF)
    for e in events:
        e.play()
    s.start()
    input("here is the goal …")
    s.stop()
    for e in events:
        e.stop()
    time.sleep(1)
    
    
    while running:
        random.shuffle(population)
        individuos = random.sample(population, 8)
        population_fitness = []
        for i in individuos:
            population_fitness.append((i, fitness2(i, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm,goalF)))
        #population_fitness = [(genome, fitness2(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm,goalF)) for genome in population]

        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=False)

        population = [e[0] for e in sorted_population_fitness]

        next_generation = population[0:2]

        if((fitness2(population[0],s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm,goalF)<0.1) or (population_id>100)):
            break

        for j in range(int(len(population) / 2) - 1):

            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            parents = selection_pair(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]

        #print(f"population {population_id} done")        
        """for i, genome in enumerate(population):
            
            print(genome_to_melody(population[0],num_bars, num_notes, num_steps, pauses,key, scale, root))
            print(genome_to_melody(population[1],num_bars, num_notes, num_steps, pauses,key, scale, root))
            print(population[0])
            print(population[1])
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
            """
        #print("done")

        population = next_generation
        population_id += 1





    events = genome_to_events(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()
    input("here is the no1 hit …")
    s.stop()
    for e in events:
        e.stop()

    time.sleep(1)

    events = genome_to_events(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()
    input("here is the second best …")
    s.stop()
    for e in events:
        e.stop()

    time.sleep(1)

    print("saving population midi …")
    #save_melody_to_midi(f"1606794889/0/aurora_theme_real.mid")
    
    save_genome_to_midi(f"resultados/resultado1.mid",population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    save_genome_to_midi(f"resultados/resultado2.mid",population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)       


if __name__ == '__main__':
    main()





