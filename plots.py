
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#

# FOUR PEAKS

fourpeaks_data_RHC = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\fourpeaks\RCH\rhc__RCH_final__curves_df.csv")
fourpeaks_data_SA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\fourpeaks\SA\sa__SA_final__curves_df.csv")
fourpeaks_data_GA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\fourpeaks\GA\ga__GA_final__curves_df.csv")
fourpeaks_data_MIM = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\fourpeaks\MIMIC\mimic__MIMIC_final__curves_df.csv")
#
plt.close()
plt.plot(fourpeaks_data_RHC["Iteration"],fourpeaks_data_RHC["Fitness"], label = "RHC")
plt.plot(fourpeaks_data_SA["Iteration"],fourpeaks_data_SA["Fitness"], label = "SA")
plt.plot(fourpeaks_data_GA["Iteration"],fourpeaks_data_GA["Fitness"], label = "GA")
plt.plot(fourpeaks_data_MIM["Iteration"],fourpeaks_data_MIM["Fitness"], label = "MIMIC")
plt.title("Four Peaks")
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig("Assignment2/charts/Four Peaks Fitness")
plt.show()

# # FLIP FLOP
flipflop_data_RHC = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\flipflop\RHC\rhc__RHC__curves_df.csv")
flipflop_data_SA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\flipflop\SA\sa__SA__curves_df.csv")
flipflop_data_GA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\flipflop\GA\ga__GA__curves_df.csv")
flipflop_data_MIM = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\flipflop\MIMIC\mimic__MIMIC__curves_df.csv")
#
plt.close()
plt.plot(flipflop_data_RHC["Iteration"],flipflop_data_RHC["Fitness"], label = "RHC")
plt.plot(flipflop_data_SA["Iteration"],flipflop_data_SA["Fitness"], label = "SA")
plt.plot(flipflop_data_GA["Iteration"],flipflop_data_GA["Fitness"], label = "GA")
plt.plot(flipflop_data_MIM["Iteration"],flipflop_data_MIM["Fitness"], label = "MIMIC")
plt.title("Flip Flop")
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig("Assignment2/charts/Flip Flop Fitness")
plt.show()


# # TSP
tsp_data_RHC = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\TSP\RHC\rhc__RHC__curves_df.csv")
tsp_data_SA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\TSP\SA\sa__SA__curves_df.csv")
tsp_data_GA = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\TSP\GA\ga__GA__curves_df.csv")
tsp_data_MIM = pd.read_csv(r"C:\Users\Wilmer\OneDrive - San Jose-Evergreen Community College District\GA Tech\CS 7641 Machine Learning\Assignment2\outputs\TSP\MIMIC\mimic__MIMIC__curves_df.csv")
#
#
plt.close()
plt.plot(tsp_data_RHC["Iteration"],tsp_data_RHC["Fitness"], label = "RHC")
plt.plot(tsp_data_SA["Iteration"],tsp_data_SA["Fitness"], label = "SA")
plt.plot(tsp_data_GA["Iteration"],tsp_data_GA["Fitness"], label = "GA")
plt.plot(tsp_data_MIM["Iteration"],tsp_data_MIM["Fitness"], label = "MIMIC")
plt.title("TSP")
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig("Assignment2/charts/TSP")
plt.show()


########################################################################################################################
#################################### Time Graphs #######################################################################
########################################################################################################################
def get_best_time(dataset, algo):
    df = dataset[dataset.Fitness == max(dataset.Fitness)]
    df = df[df.Time == min(df.Time)]
    df['Algo'] = algo
    return df

# FOUR PEAKS
fourpeaks_time_RHC = get_best_time(fourpeaks_data_RHC, "RHC")
fourpeaks_time_SA = get_best_time(fourpeaks_data_SA, "SA")
fourpeaks_time_GA = get_best_time(fourpeaks_data_GA, "GA")
fourpeaks_time_MIM = get_best_time(fourpeaks_data_MIM, "MIM")

plt.close()
plt.bar(fourpeaks_time_RHC["Algo"], fourpeaks_time_RHC["Time"])
plt.bar(fourpeaks_time_SA["Algo"],fourpeaks_time_SA["Time"])
plt.bar(fourpeaks_time_GA["Algo"],fourpeaks_time_GA["Time"])
plt.bar(fourpeaks_time_MIM["Algo"],fourpeaks_time_MIM["Time"])
plt.title("Time - Four Peaks")
plt.xlabel('Algorithm')
plt.ylabel('Time')
plt.savefig("Assignment2/charts/Time- Four Peaks")
plt.show()


# FLIP FLOP
flipflop_time_RHC = get_best_time(flipflop_data_RHC, "RHC")
flipflop_time_SA = get_best_time(flipflop_data_SA, "SA")
flipflop_time_GA = get_best_time(flipflop_data_GA, "GA")
flipflop_time_MIM = get_best_time(flipflop_data_MIM, "MIM")

plt.close()
plt.bar(flipflop_time_RHC["Algo"], flipflop_time_RHC["Time"])
plt.bar(flipflop_time_SA["Algo"],flipflop_time_SA["Time"])
plt.bar(flipflop_time_GA["Algo"],flipflop_time_GA["Time"])
plt.bar(flipflop_time_MIM["Algo"],flipflop_time_MIM["Time"])
plt.title("Time - Flip Flop")
plt.xlabel('Algorithm')
plt.ylabel('Time')
plt.savefig("Assignment2/charts/Time- Flip Flop")
plt.show()

# TSP
tsp_time_RHC = get_best_time(tsp_data_RHC, "RHC")
tsp_time_SA = get_best_time(tsp_data_SA, "SA")
tsp_time_GA = get_best_time(tsp_data_GA, "GA")
tsp_time_MIM = get_best_time(tsp_data_MIM, "MIM")

plt.close()
plt.bar(tsp_time_RHC["Algo"], tsp_time_RHC["Time"])
plt.bar(tsp_time_SA["Algo"],tsp_time_SA["Time"])
plt.bar(tsp_time_GA["Algo"],tsp_time_GA["Time"])
plt.bar(tsp_time_MIM["Algo"],tsp_time_MIM["Time"])
plt.title("Time - Traveling Salesman Problem")
plt.xlabel('Algorithm')
plt.ylabel('Time')
plt.savefig("Assignment2/charts/Time- TSP")
plt.show()
