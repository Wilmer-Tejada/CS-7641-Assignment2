import mlrose_hiive as mlrose
import numpy as np

fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=800, fitness_fn=fitness, maximize=True, max_val=2)

# RHC - Done
rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RCH",
                       output_directory=r"Assignment2\outputs\fourpeaks",
                       seed=42,
                       iteration_list=100 * np.arange(9),
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()

# SA - Done
sa = mlrose.SARunner(problem=problem,
                     experiment_name="SA",
                     output_directory=r"Assignment2\outputs\fourpeaks",
                     seed=42,
                     iteration_list=100 * np.arange(9),
                     max_attempts=1000,
                     # temperature_list=[1],
                     temperature_list=[1],
                     decay_list=[mlrose.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()

# GA - Done
ga = mlrose.GARunner(problem=problem,
                     experiment_name="GA",
                     output_directory=r"Assignment2\outputs\fourpeaks",
                     seed=42,
                     iteration_list= 100 * np.arange(9),
                     max_attempts=1000,
                     # population_sizes=[150],
                     population_sizes=[300],
                     # mutation_rates=[0.4],
                     mutation_rates=[0.6]
                     )
ga_run_stats, ga_run_curves = ga.run()


mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory=r'Assignment2\outputs\fourpeaks',
                           seed=None,
                           iteration_list=100 * np.arange(9),
                           population_sizes=[100],
                           max_attempts=800,
                           keep_percent_list=[0.5],
                           # keep_percent_list=[0.25, 0.5, 0.75],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()






