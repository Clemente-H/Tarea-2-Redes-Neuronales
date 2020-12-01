# genetic algorithm to find music?
Para el desarrollo de esta tarea, se plantea la pregunta si dado una cancion, se puede generar un algoritmo que cree una cancion similar.
Intentando acercarse lo mas posible a esta







## How to use this script

To start the script call:

```
$ python mgen.py
```

The script will ask you to define couple parameters:

| Population Size | Number of melodies per generation to rate and recombine
| Number of mutations | Max number of mutations that should be possible per child generated
| Mutation probability | Probability for a mutation to occur

After you defined all parameters above the genetic algorithm will generate a population of melodies and play each one back to you. After each playback you can rate the melody. 

Each generation all melodies are saved in midi format to disk using the following system:
```
<timestamp>/<generation>/<rating>.mid
```

## Contribution



Have a great day Coders!
