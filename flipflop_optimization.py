import numpy as np
import mlrose_hiive as mlrose

fitness = mlrose.FlipFlop()
problem = mlrose.DiscreteOpt(length=800, fitness_fn=fitness, maximize=True, max_val=2)

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RHC",
                       output_directory=r"Assignment2\outputs\flipflop",
                       seed=42,
                       iteration_list=100 * np.arange(9),
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()

# SA - Done
sa = mlrose.SARunner(problem=problem,
                     experiment_name="SA",
                     output_directory=r"Assignment2\outputs\flipflop",
                     seed=42,
                     iteration_list=100 * np.arange(9),
                     max_attempts=1000,
                     temperature_list=[1], #
                     # temperature_list=[1, 50, 100, 250, 1000],
                     decay_list=[mlrose.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()

# GA - Done
ga = mlrose.GARunner(problem=problem,
                     experiment_name="GA",
                     output_directory=r"Assignment2\outputs\flipflop",
                     seed=42,
                     iteration_list= 100 * np.arange(9),
                     max_attempts=1000,
                     # population_sizes=[200],
                     population_sizes=[300],
                     # mutation_rates=[0.6])
                     mutation_rates=[0.4]
                     )
ga_run_stats, ga_run_curves = ga.run()

# Still need to run this to figure out how to not have a flat curve
mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory=r'Assignment2\outputs\flipflop',
                           seed=42,
                           iteration_list=100 * np.arange(9),
                           population_sizes=[300],
                           max_attempts=500,
                           # keep_percent_list=[0.25, 0.5, 0.75],
                           keep_percent_list=[0.9],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
