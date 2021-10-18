import numpy as np
import mlrose_hiive as mlrose


problem = mlrose.TSPGenerator().generate(seed=42, number_of_cities=20)
problem.maximize = 1

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RHC",
                       output_directory=r"Assignment2\outputs\TSP",
                       seed=123456,
                       iteration_list=100 * np.arange(9),
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()
# # #
# # # # SA - Done
sa = mlrose.SARunner(problem=problem,
                     experiment_name="SA",
                     output_directory=r"Assignment2\outputs\TSP",
                     seed=42,
                     iteration_list=100 * np.arange(9),
                     max_attempts=1000,
                     temperature_list=[100], #
                     # temperature_list=[.5, 1,20, 50, 100, 1000, 10000],
                     decay_list=[mlrose.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()
# #
# # GA - Done
ga = mlrose.GARunner(problem=problem,
                     experiment_name="GA",
                     output_directory=r"Assignment2\outputs\TSP",
                     seed=42,
                     iteration_list= 100 * np.arange(9),
                     max_attempts=1000,
                     population_sizes=[150],
                     # population_sizes=[150,200,300],
                     mutation_rates=[0.4]
                     # mutation_rates=[0.4, 0.5, 0.6]
                     )
ga_run_stats, ga_run_curves = ga.run()

mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory=r'Assignment2\outputs\TSP',
                           seed=42,
                           iteration_list=100 * np.arange(9),
                           population_sizes=[2000], # 1000 works best so far
                           max_attempts=500,
                           keep_percent_list=[0.5],
                           # keep_percent_list=[0.9],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
