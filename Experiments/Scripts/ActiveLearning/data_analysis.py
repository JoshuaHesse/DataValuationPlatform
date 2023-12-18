#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:38:31 2023

@author: joshua
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
from scipy import stats
from typing import List

def plot_learning(combined_names: List[str],
                  all_influence_actives: List[List[np.ndarray]],
                  greedy_actives:List[np.ndarray],
                  random_actives:List[np.ndarray],
                  y_p_train: np.ndarray,
                  filename: str,
                  name: str,
                  influence: str,
                  step_size :float = 1,
                  steps :int = 5):
    """


    Parameters
    ----------
    combined_names : List[str]; Len = N
        List of strings made up by combining the influence-, regression-, and
        sampling-strategy. The order of names should correspond to the
        order in all_influences_actives
    all_influence_actives : List[List[np.ndarray]]; Len = N, inner list: len = replicates
        List of Lists of replicates of np.ndarrays.containing the actives found for each corresponding
        combinartion of influence-, regression-, and sampling-strategies.
    greedy_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using 
        the greedy approach.
    random_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using the 
        random sampling approach.
    y_p_train : np.ndarray
        Activity labels (1 or 0) for the current dataset.
    filename : str
        Name of the folder the results should be saved in.
    name : str
        Name of the current dataset.
    influence : str
        Name of the influence function used.
    step_size: float, optional
        Size of the steps as a fraction of the original dataset size. The
        Default is 1
    steps: int, optional
        Number of iterations for active learning. The Default is 5.
        

    Returns
    -------
    None.

    -------

    This functions plots all combinations of influence-, regression-, and
    sampling-strategies, that were calculated for the given dataset. Additionally,
    the plot contains the greedy approach as well as a ramdom sampling approach.
    All actives_found values are normilized to 0-100%. The results are plotted as 
    means, with their standard deviation as error bars.
    """
    #steps are the additional steps after the initial random batch
    #therefore, the position 0 with 0 actives, as well as the first 
    #random batch have to be added to the plot --> steps+2
    steps = steps + 2

    #y_axis labels
    sampling_percentages = [step_size * t for t in range(steps)]

    plt.figure(figsize=(12, 6))
    
    #the results are read from the lists and converted to np.arrays
    #to allow easy mean and std calculations
    greedy_results = np.zeros((steps, len(greedy_actives)))
    random_results = np.zeros((steps, len(random_actives)))
    for i in range(len(greedy_actives)):
        greedy_results[1:,i] = greedy_actives[i]
        random_results[1:,i] = random_actives[i]


    greedy_means = np.mean(greedy_results, axis=1)
    greedy_stds = np.std(greedy_results, axis=1)
    random_means = np.mean(random_results, axis=1)
    random_stds = np.std(random_results, axis=1)

    #plotting of greedy and random results
    plt.errorbar(sampling_percentages, greedy_means, yerr=greedy_stds, marker='o', label='Greedy Aquisition', color='grey', capsize=5)
    plt.errorbar(sampling_percentages, random_means, yerr=random_stds, marker='o', label='Random Performance', color='black', capsize=5)


    #the influence results are read from the lists and converted to np.arrays
    #for ease of mean&std calculation, and plotted.
    for method in range(len(combined_names)):
        current_results = np.zeros((steps, len(all_influence_actives[method])))
        for replicate in range(len(all_influence_actives[method])):
            current_results[1:,replicate] = all_influence_actives[method][replicate]
  
        current_means = np.mean(current_results, axis=1)
        current_stds = np.std(current_results, axis=1)
        plt.errorbar(sampling_percentages, current_means, yerr=current_stds, marker='o', label=combined_names[method], capsize=5)


    plt.xlabel('Sampling Percentage')
    plt.ylabel('Percentage of Active Samples')
    plt.title("Active learning influence vs greedy on " + name)
    plt.legend(loc='upper left')

    path = "../../Results/ActiveLearning/" + filename + "/" + name + "/" + influence + ".png"
    print("----------------------")
    print("[eval]: Results saved in " + path)

    #the plot is saved to the folder defined as filename in the pipeline^
    plt.savefig(path)



def plot_learning_bands(combined_names: List[str],
                  all_influence_actives: List[List[np.ndarray]],
                  greedy_actives:List[np.ndarray],
                  random_actives:List[np.ndarray],
                  y_p_train: np.ndarray,
                  filename: str,
                  name: str,
                  influence: str,
                  step_size = 1,
                  steps = 5):
    """


    Parameters
    ----------
    combined_names : List[str]; Len = N
        List of strings made up by combining the influence-, regression-, and
        sampling-strategy. The order of names should correspond to the
        order in all_influences_actives
    all_influence_actives : List[List[np.ndarray]]; Len = N, inner list: len = replicates
        List of Lists of replicates of np.ndarrays.containing the actives found for each corresponding
        combinartion of influence-, regression-, and sampling-strategies.
    greedy_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using 
        the greedy approach.
    random_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using the 
        random sampling approach.
    y_p_train : np.ndarray
        Activity labels (1 or 0) for the current dataset.
    filename : str
        Name of the folder the results should be saved in.
    name : str
        Name of the current dataset.
    influence : str
        Name of the influence function used.
    step_size: float, optional
        Size of the steps as a fraction of the original dataset size. The
        Default is 1
    steps: int, optional
        Number of iterations for active learning. The Default is 5.
        
    Returns
    -------
    None.

    -------

    This functions plots all combinations of influence-, regression-, and
    sampling-strategies, that were calculated for the given dataset. Additionally,
    the plot contains the greedy approach as well as a ramdom sampling approach.
    All actives_found values are normilized to 0-100%. The results are plotted
    as means, with the standard deviation plotted as error bands.
      """
    #steps are the additional steps after the initial random batch
    #therefore, the position 0 with 0 actives, as well as the first 
    #random batch have to be added to the plot --> steps+2
    steps = steps + 2

    #y_axis labels
    sampling_percentages = [step_size * t for t in range(steps)]

    plt.figure(figsize=(12, 6))
       
    #the results are read from the lists and converted to np.arrays
    #to allow easy mean and std calculations
    greedy_results = np.zeros((steps, len(greedy_actives)))
    random_results = np.zeros((steps, len(random_actives)))
    for i in range(len(greedy_actives)):
        greedy_results[1:,i] = greedy_actives[i]
        random_results[1:,i] = random_actives[i]

    greedy_means = np.mean(greedy_results, axis=1)
    greedy_stds = np.std(greedy_results, axis=1)
    random_means = np.mean(random_results, axis=1)
    random_stds = np.std(random_results, axis=1)

    plt.plot(sampling_percentages, greedy_means, label='Greedy Aqcuisition', color = "grey")
    plt.fill_between(sampling_percentages, greedy_means - greedy_stds, greedy_means + greedy_stds, alpha=0.2)
    
    plt.plot(sampling_percentages, random_means, label='Random Sampling', color = "black")
    plt.fill_between(sampling_percentages, random_means - random_stds, random_means + random_stds, alpha=0.2)


    #the influence results are read from the lists and converted to np.arrays
    #for ease of mean&std calculation, and plotted.
    for method in range(len(combined_names)):
        current_results = np.zeros((steps, len(all_influence_actives[method])))
        for replicate in range(len(all_influence_actives[method])):
            current_results[1:,replicate] = all_influence_actives[method][replicate]
  
        current_means = np.mean(current_results, axis=1)
        current_stds = np.std(current_results, axis=1)
        plt.plot(sampling_percentages, current_means, label=combined_names[method])
        plt.fill_between(sampling_percentages, current_means - current_stds, current_means + current_stds, alpha=0.2)

    plt.xlabel('Sampling Percentage')
    plt.ylabel('Percentage of Active Samples')
    plt.title("Active learning influence vs greedy on " + name)
    plt.legend(loc='upper left')

    path = "../../Results/ActiveLearning/" + filename + "/" + name + "/" + influence + "_bands.png"
    print("----------------------")
    print("[eval]: Results saved in " + path)

    #the plot is saved to the folder defined as filename in the pipeline^
    plt.savefig(path)

def save_data(combined_names: List[str],
              all_influence_actives: List[List[np.ndarray]],
              greedy_actives: List[np.ndarray], 
              random_actives:List[np.ndarray],
              y_p_train: np.ndarray,
              times: List[List[float]],
              greedy_time: List[float],
              random_time:List[float],
              filename: str,
              name: str,
              influence: str,
              steps:int = 5,
              step_size:float = 1,
              replicates: int = 5):
   """
    

    
    Parameters
    ----------
    combined_names : List[str]; Len = N
        List of strings made up by combining the influence-, regression-, and
        sampling-strategy. The order of names should correspond to the
        order in all_influences_actives
    all_influence_actives : List[List[np.ndarray]]; Len = N, inner list: len = replicates
        List of Lists of replicates of np.ndarrays.containing the actives found for each corresponding
        combinartion of influence-, regression-, and sampling-strategies.
    greedy_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using 
        the greedy approach.
    random_actives : List[np.ndarray]
        List of replicates of arrays containing the actives found when using the 
        random sampling approach.
    y_p_train : np.ndarray
        Activity labels (1 or 0) for the current dataset.
    times: List[List[float]]
        List of Lists of replicates of floats, representing the computation time
        of each replicate for the influence functions
    greedy_time: List[float]
        List of floats representing the computational time taken by the greedy 
        approach replicates
    random_time: List[float]
        List of floats representing the computational time taken by the random 
        sampling approach replicates
    filename : str
        Name of the folder the results should be saved in.
    name : str
        Name of the current dataset.
    influence : str
        Name of the influence function used.
    steps: int, optional
        Number of iterations for active learning. The default is 5.
    step_size. float, optional,
        Size of each fraction of the dataset taken during active learning in
        percent of length of the original training dataset. The default is 1.
    replicates: int, optional
        Number of replicates. The default is 5.
    Returns
    -------
    None.
    
    ------
    
    For a given datast, this function collects actives_found for all combinations
    of influence-, regression-, and sampling-strategies, as well as the greedy 
    actives found and the actives found by random sampling, and exports the 
    data into a csv file.
    

   """
   #as the steps represent the additional steps after the initial random
   #split, the results contain 1 additional datapoint 
   steps = steps + 1

   #greedy and random results are added to the list of results to allow
   #easy exporting
   combined_names.append("Greedy Acquisition")
   times.append(greedy_time)
   combined_names.append("Random Sampling")
   times.append(random_time)
   all_influence_actives.append(greedy_actives)
   all_influence_actives.append(random_actives)
   #generate a dictionary to link each np.ndarray within all_influences_actives
   #with its corresponding label in combined_names

   data_replicates = []
   for rep in range(replicates):
       tmp = {name: result[rep] for name, result in zip(combined_names, all_influence_actives)}
       data_replicates.append(pd.DataFrame(tmp, index = range(steps)))
   
   #all replicates are concatenated to one final dataframe.
   #between each replicate, an empty row is inserted to allow 
   #easy discrimination.
       
    # Create a dataframe with a single row of NaN values
   empty_df = pd.DataFrame([np.nan]*len(combined_names), index=combined_names).T

    # Concatenate each dataframe with the empty dataframe
   summary = pd.concat([pd.concat([df, empty_df]) for df in data_replicates])

# Do the same for summary_times

   path = "../../Results/ActiveLearning/" + filename + "/" + name + "/" + influence + ".csv"
   #save final dataframe in the filename folder set in the pipeline´
   summary.to_csv(path)
   print("[eval]: Results saved in " + path)
    
    
   #the calculations times are exported the same way as was done for the results
   times_replicates = []
   for rep in range(replicates):
       tmp = {name: time[rep] for name, time in zip(combined_names, times)}
       times_replicates.append(pd.DataFrame(tmp, index = [0]))
   summary_times = pd.concat([pd.concat([df, empty_df]) for df in times_replicates])
 
   path = "../../Results/ActiveLearning/" + filename + "/" + name + "/" + influence + "_time.csv"
   #save dataframe in the filename folder set in the pipeline´
   summary_times.to_csv(path)
   print("[eval]: Results saved in " + path)



#################################################################################
#FP TP plot precision, bedroc, EF10, and time
#################################################################################





def find_average_pvalue(reference_dataframe, method_dataframe, metric, replicates):
    
    columns = [metric +" - Replicate " + str(i+1) for i in range(replicates)]

    method_metric_replicates_and_datasets = method_dataframe.loc[:,columns]
    method_metric_replicates = method_metric_replicates_and_datasets.mean(axis=0)
    score_metric_datasets = reference_dataframe.loc[:,metric +" - mean"]
    score_metric_mean = score_metric_datasets.mean(axis=0)
    method_minus_score = method_metric_replicates - score_metric_mean
    result = stats.wilcoxon(x = method_minus_score, alternative = "greater")
    return result[1]

def find_dataset_pvalue(reference_dataframe, method_dataframe, metric, replicates):
    columns = [metric + " - Replicate " + str(i+1) for i in range(replicates)]
    
    method_metric_replicates_and_datasets = method_dataframe.loc[:, columns]
    score_metric_datasets = reference_dataframe.loc[:, metric + " - mean"]
    
    p_values = []
    for row in range(len(method_metric_replicates_and_datasets)):
        method_replicates = method_metric_replicates_and_datasets.loc[row]
        method_minus_score = method_replicates - score_metric_datasets[row]
        if method_minus_score.sum()!=0:
            result = stats.wilcoxon(x=method_minus_score, alternative="greater")
            p_values.append(result[1])
        else:
            p_values.append(1)
    return p_values
    

        
def score_and_filter_processing(df, metric):
    metric_datasets = df.loc[:,metric +" - mean"]

    if metric == "FP Precision@90":
        fp_rates = df.loc[:,"FP rate"] 
        metric_datasets = metric_datasets.sub(fp_rates, axis=0)
    elif metric == "TP Precision@90":
        tp_rates = df.loc[:,"TP rate"] 
        metric_datasets = metric_datasets.sub(tp_rates, axis=0)

    return metric_datasets.mean(axis=0)

def plot_average_results_combined(score_dataframe, pains_dataframe, method_dataframe_list, method_names, metric, replicates, ax):
    columns = [metric +" - Replicate " + str(i+1) for i in range(replicates)]

    significance_level = 0.05/len(score_dataframe)
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]

    score_metric_mean = score_and_filter_processing(score_dataframe, metric)
    filter_metric_mean = score_and_filter_processing(pains_dataframe, metric)

    if metric in ["FP Precision@90", "FP EF10", "FP BEDROC20"]:
        ax.bar("Fragment Filter",filter_metric_mean, capsize=4, color = "#696f78")
    elif metric in  ["TP Precision@90", "TP EF10", "TP BEDROC20"]:
        ax.bar("Score",score_metric_mean, capsize=4, color = '#A9A9A9')
    highest_star = 0

    for method in range(len(method_names)):
        method_dataframe = method_dataframe_list[method]
        if metric in ["FP Precision@90", "FP EF10", "FP BEDROC20"]:
            p_value = find_average_pvalue(pains_dataframe, method_dataframe, metric, replicates)
        elif metric in  ["TP Precision@90", "TP EF10", "TP BEDROC20"]:
            p_value = find_average_pvalue(score_dataframe, method_dataframe, metric, replicates)

        method_metric_replicates_and_datasets = method_dataframe.loc[:,columns]

        if metric == "FP Precision@90":
            fp_rates = method_dataframe.loc[:,"FP rate"]
            method_metric_replicates_and_datasets = method_metric_replicates_and_datasets.sub(fp_rates, axis=0)
        elif metric == "TP Precision@90":
            tp_rates = method_dataframe.loc[:,"TP rate"]
            method_metric_replicates_and_datasets = method_metric_replicates_and_datasets.sub(tp_rates, axis=0)
        method_metric_replicates = method_metric_replicates_and_datasets.mean(axis=0)
        method_mean = method_metric_replicates.mean()
        method_std = method_metric_replicates.std()
        ax.bar(method_names[method],method_mean, yerr=method_std, capsize=8, color = colors[method])
        # Initialize a variable to store the y-coordinate of the highest star


        # Calculate the y-coordinate of the star
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        five_percent_y = 0.05 * yrange
        star_height = method_mean + method_std + five_percent_y

        # Update the y-coordinate of the highest star
        highest_star = max(highest_star, star_height)

        # Add the star annotation
        if p_value < significance_level:
            ax.text(method_names[method], star_height, '*', ha='center', va='bottom', fontsize=16)

    # After all the bars and stars have been plotted, set the y-limit
    # Check if the highest star is outside the current y limit
    if highest_star > ymax:
        # if it is, update the upper y limit to the highest star
        ax.set_ylim((ymin, highest_star + five_percent_y*2))  # added a little extra space above the highest star

    ax.axhline(y=0, linestyle='--', color='black', alpha=0.6)
    ax.set_title("Average " + metric + " over all datasets")
    #ax.set_xlabel('Prediction Methods')
    if metric not in ["FP Precision@90", "TP Precision@90"]:
        ax.set_ylabel(' Performance ' + metric)
    else:
        ax.set_ylabel('Relative Performance ' + metric)
    ax.tick_params(axis='x', rotation=45, labelsize='small')
    #ax.set_xticklabels(labels, rotation=45, ha='right')

def plot_all_dataset_dots_combined(results_score, results_pains, method_dataframe_list, method_names, datasets, metric, replicates, ax):
    columns = [metric +" - Replicate " + str(i+1) for i in range(replicates)]
    results_mean = []
    results_std = []
    p_values = []  # New list to store p-values

    for i in range(len(method_names)):
        method_dataframe = method_dataframe_list[i]
        if metric in ["FP Precision@90", "FP EF10", "FP BEDROC20"]:
            reference_results = results_pains
        elif metric in  ["TP Precision@90", "TP EF10", "TP BEDROC20"]:
            reference_results = results_score

        method_metric_replicates_and_datasets = method_dataframe.loc[:, columns]
        method_metric_datasets_mean = method_metric_replicates_and_datasets.mean(axis=1)
        method_metric_datasets_std = method_metric_replicates_and_datasets.std(axis=1)
        results_mean.append(method_metric_datasets_mean)
        results_std.append(method_metric_datasets_std)

        # Calculate p-values for each dataset and method against the reference method
        p_values.append(find_dataset_pvalue(reference_results, method_dataframe, metric, replicates))

    results_mean_df = np.array(pd.concat(results_mean, axis=1))
    results_std_df = np.array(pd.concat(results_std, axis=1))
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='grey', alpha =0.3)

    for i, dataset in enumerate(datasets):
        ax.scatter(x=reference_results.loc[i, metric + " - mean"], y=i, c="#696f78")

        for j in range(len(method_names)):
            # Add dark outline for significant methods
            if p_values[j][i] < 0.002:
                scatter = ax.scatter(x=results_mean_df[i][j], y=i, c=colors[j], edgecolors='black', linewidths=1)
            else:
                scatter = ax.scatter(x=results_mean_df[i][j], y=i, c=colors[j])
    ax.set_title(metric + " for each dataset")
    ax.set_xlabel('Performance '+ metric)
    ax.set_ylabel('Datasets')

    ax.set_yticks(range(25))
    ax.set_yticklabels(datasets)

    # Remove y-axis text for the second and third columns
    if ax.get_subplotspec().colspan.start in [1, 2]:
        ax.set_yticklabels([])

    plt.tick_params(axis='y', labelsize=7)

    plt.subplots_adjust(top=1)

    # Returning the p-values for future use
    return p_values




def combined_plot(score_dataframe, pains_dataframe, method_dataframe_list, methods_names, metric_list, replicates, filename):
    names = methods_names
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    datasets =  score_dataframe.loc[:, "Unnamed: 0"].tolist()
    # Assuming the metrics are given in the order of ["FP Precision@90", "FP EF10", "FP BEDROC20"]
    for i, metric in enumerate(metric_list):
        plot_average_results_combined(score_dataframe, pains_dataframe, method_dataframe_list, names, metric, replicates, axs[0, i])
        plot_all_dataset_dots_combined(score_dataframe, pains_dataframe, method_dataframe_list, names, datasets, metric, replicates, axs[1, i])

    # Create Lines for the legend
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]
    if "FP Precision@90" in metric_list:
        colors.insert(0,"#696f78")
        names.insert(0,"Fragment Filter")
    else:
        colors.insert(0,"#A9A9A9")
        names.insert(0,"Score")
    legend_handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None') for c in colors]

    plt.tight_layout()
    # Place the legend outside plot area, above
    fig.legend(legend_handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.05))
    path = "../Results/images/" + filename + ".png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_average_results_time(score_dataframe, pains_dataframe, method_dataframe_list, method_names, metric, replicates, ax):
    columns = [metric +" - Replicate " + str(i+1) for i in range(replicates)]

    significance_level = 0.05/len(score_dataframe)
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]

    score_metric_mean = score_and_filter_processing(score_dataframe, metric)
    filter_metric_mean = score_and_filter_processing(pains_dataframe, metric)

    ax.bar("Filter",filter_metric_mean, capsize=4, color = "#696f78")

    ax.bar("Score",score_metric_mean, capsize=4, color = '#A9A9A9')


    for method in range(len(method_names)):
        method_dataframe = method_dataframe_list[method]


        method_metric_replicates_and_datasets = method_dataframe.loc[:,columns]


        method_metric_replicates = method_metric_replicates_and_datasets.mean(axis=0)
        method_mean = method_metric_replicates.mean()
        method_std = method_metric_replicates.std()
        ax.bar(method_names[method],method_mean, yerr=method_std, capsize=8, color = colors[method])
        # Initialize a variable to store the y-coordinate of the highest star

    ax.axhline(y=0, linestyle='--', color='black', alpha=0.6)
    ax.set_title("Average " + metric + " over all datasets")
    #ax.set_xlabel('Prediction Methods')
    ax.set_ylabel('Average time per replicate in [s]')
    ax.tick_params(axis='x', rotation=45, labelsize='small')
    #ax.set_xticklabels(labels, rotation=45, ha='right')

def plot_all_dataset_dots_time(results_score, results_pains, method_dataframe_list, method_names, datasets, metric, replicates, ax):
    columns = [metric +" - Replicate " + str(i+1) for i in range(replicates)]
    results_mean = []
    results_std = []
    p_values = []  # New list to store p-values

    for i in range(len(method_names)):
        method_dataframe = method_dataframe_list[i]

        reference_results = results_pains

        reference_results = results_score

        method_metric_replicates_and_datasets = method_dataframe.loc[:, columns]
        method_metric_datasets_mean = method_metric_replicates_and_datasets.mean(axis=1)
        method_metric_datasets_std = method_metric_replicates_and_datasets.std(axis=1)
        results_mean.append(method_metric_datasets_mean)
        results_std.append(method_metric_datasets_std)



    results_mean_df = np.array(pd.concat(results_mean, axis=1))
    results_std_df = np.array(pd.concat(results_std, axis=1))
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='grey', alpha =0.3)

    for i, dataset in enumerate(datasets):
        ax.scatter(x=results_pains.loc[i, metric + " - mean"], y=i, c="#696f78")
        ax.scatter(x=results_score.loc[i, metric + " - mean"], y=i, c="#A9A9A9")
        for j in range(len(method_names)):
            # Add dark outline for significant methods
            scatter = ax.scatter(x=results_mean_df[i][j], y=i, c=colors[j])
    ax.set_title(metric + " for each dataset")
    ax.set_xlabel('Time per replicate in [s]')


    ax.set_yticks(range(25))
    ax.set_yticklabels(datasets)

    # Remove y-axis text for the second and third columns


    plt.tick_params(axis='y', labelsize=7)

    plt.subplots_adjust(top=1)



def combined_plot_time(score_dataframe, pains_dataframe, method_dataframe_list, methods_names, metric_list, replicates, filename):
    names = methods_names
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    datasets =  score_dataframe.loc[:, "Unnamed: 0"].tolist()
    for i, metric in enumerate(metric_list):
        plot_average_results_time(score_dataframe, pains_dataframe, method_dataframe_list, names, metric, replicates, axs[0])
        plot_all_dataset_dots_time(score_dataframe, pains_dataframe, method_dataframe_list, names, datasets, metric, replicates, axs[1])

    # Create Lines for the legend
    colors = ["#003f5c","#58508d","#bc5090","#d784be","#ffd0ff","#f95d6a","#ff8d95","#ffa600"]

    colors.insert(0,"#696f78")
    names.insert(0,"Filter")

    colors.insert(0,"#A9A9A9")
    names.insert(0,"Score")
    legend_handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None') for c in colors]

    plt.tight_layout()
    # Place the legend outside plot area, above
    fig.legend(legend_handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.05))
    path = "../../Results/ActiveLearning/" + filename + ".png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()











