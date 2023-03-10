###Author Marco Sanna (marco-sanna) 02/03/2023
#Defines a class for statistical analysis, figure visualization and machine learning 

import pandas as pd
import numpy as np
from functools import reduce
import plotly.io as pi
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from math import sqrt
from math import log2
from math import isnan
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib.transforms
from adjustText import adjust_text
import seaborn as sns
import re
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.transforms as transforms
from statannotations.Annotator import Annotator
from matplotlib.patches import PathPatch
from statannot import add_stat_annotation
import sklearn
from sklearn import metrics
import pingouin as pg
import pickle

def multiple_comparisons_correction(results, correction):
    """
    Perform multiple comparisons correction on a list of p-values.

    Parameters:
    - results: list of tuples, where each tuple is of the form (test name, p-value)
    - correction: str, either 'bonferroni' or 'holm-bonferroni'

    Returns:
    - corrected_results: list of tuples, where each tuple is of the form (test name, corrected p-value)
    """
    n = len(results)
    p_values = [p for _, _, p in results]

    if correction == 'bonferroni':
        corrected_p_values = [min(p * n, 1) for p in p_values]

    elif correction == 'holm-bonferroni':
        corrected_p_values = [min(p * n / (i + 1), 1)  for i, p in enumerate(sorted(p_values, reverse=True))]

    corrected_results = [(i, test, p) for (i, test, _), p in zip(results, corrected_p_values)]

    return corrected_results

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic
    dfn = x.size-1 #define degrees of freedom numerator
    dfd = y.size-1 #define degrees of freedom denominator
    p = 1-stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic
    return f, p

def variance_test(g1, g2, val):
    f, p = f_test(g1[val], g2[val])
    return p < 0.05

def var_test(g1, g2, val):
  stat, p = stats.levene(g1[val].dropna(), g2[val].dropna())
  return p < 0.05

def confidence_interval(data, alpha=0.05):
    n = len(data.dropna())
    mean = data.mean()
    std_err = data.sem()
    h = 2*std_err
    lower = mean - h
    upper = mean + h
    return n, mean, std_err, lower, upper

def norm_test(g1, g2, val, test):
  if test == "KS":
    stat, p = stats.ks_2samp(g1[val].dropna(), g2[val].dropna())
    return p > 0.05
  elif test == "DAP":
    stat1, p1 = stats.normaltest(g1[val].dropna())
    stat2, p2 = stats.normaltest(g2[val].dropna())
    return p1 and p2 > 0.05
  elif test == "SW":
    stat1, p1 = stats.shapiro(g1[val].dropna())
    stat2, p2 = stats.shapiro(g2[val].dropna())
    return p1 and p2 > 0.05

def unify_ab_values(ab_vals):
  n1 = list(ab_vals).count(1)
  n2 = list(ab_vals).count(0.5)
  n3 = list(ab_vals).count(0)

  return (1 * n1 + 0.5 * n2) / len(ab_vals)

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def dict_result_no_filter(d1, k, test_name, test_stat, g1_n, g2_n, g1_m, g2_m, start1, start2, end1, end2, g1_s, g2_s, fc):
  d1[k] = {"N G1": round(g1_n, 2),
           "Mean G1": round(g1_m, 2),
           "SD G1": round(g1_s, 2),
           "CI G1": f"[{round(start1, 2)}, {round(end1,2)}]",
           "N G2": round(g2_n, 2),
           "Mean G2": round(g2_m, 2),
           "SD G2": round(g2_s, 2),
           "CI G2": f"[{round(start2, 2)}, {round(end2, 2)}]",
           "log2FC": round(log2(fc), 2),
           "Test": test_name,
           "test_stat": round(test_stat.statistic, 2),
           "p-value" : np.format_float_scientific(test_stat.pvalue, 2)}

def binary_performances(y_true, y_prob, thresh=0.5, labels=['Positives',"Negatives"], outName = None):

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, auc, roc_curve

    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError('A binary class problem is required')
        else:
            y_prob = y_prob[:,1]

    plt.figure(figsize=[15,4])

    #1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob>thresh).astype(int))

    plt.subplot(131)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                     annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives',
               'False Negatives', 'True Positives']
    for i,t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])
    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    #2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(132)
    plt.hist(y_prob[y_true==1], density=True, bins=25,
             alpha=.5, color='green',  label=labels[0])
    plt.hist(y_prob[y_true==0], density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(thresh, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0,1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    #3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(fp_rates, tp_rates, color='orange',
             lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)
    
    plt.savefig(outName, dpi = 600)
    plt.show()

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2*(precision * recall) / (precision + recall)
    results = {
        "Precision": precision, "Recall": recall,
        "F1 Score": F1, "AUC": roc_auc
    }

    prints = [f"{kpi}: {round(score, 3)}" for kpi,score in results.items()]
    prints = ' | '.join(prints)
    print(prints)
  

class Analyzer():
  def __init__(self, file_path, meta_cols, cat_cols):
    self.df = pd.read_excel(file_path)
    self.meta_cols = meta_cols
    self.cat_cols = cat_cols
    
    #Vaccine Response Classification
    self.feats_resp = ["TT Response", "Measles Response"]
    self.time_points = time_points = ["entry", 2., 5., 9., 10., 18., 19.]
    
    #Division between patients who received an extra measle vaccination ad those who didn't
    self.nog3_df = self.df[self.df["EXTRA VACCINATION"] == 0].reset_index(drop = True)
    self.g3_df = self.df[self.df["EXTRA VACCINATION"] == 1].reset_index(drop = True)
    
    #Division of database by Groups in analysis: 
    #HEI
    self.hei_df = self.df[self.df["Group"] == "HEI"].reset_index(drop = True)
    #HEI filtered of extra measle vaccinacion
    self.hei_nog3_df = self.nog3_df[self.nog3_df["Group"] == "HEI"].reset_index(drop = True)
    #HEI and presence of extra measle vaccinacion
    self.hei_g3_df = self.g3_df[self.g3_df["Group"] == "HEI"].reset_index(drop = True)
    #HEU
    self.heu_df = self.df[self.df["Group"] == "HEU"].reset_index(drop = True)
    #HEU filtered of extra measle vaccinacion
    self.heu_nog3_df = self.nog3_df[self.nog3_df["Group"] == "HEU"].reset_index(drop = True)
    #HEU and presence of extra measle vaccinacion
    self.heu_g3_df = self.g3_df[self.g3_df["Group"] == "HEU"].reset_index(drop = True)
    #HUU
    self.cs_df = self.df[self.df["Group"] == "CS"].reset_index(drop = True)
  
  #Define columns containing metadata information
  def metadata(self):
    return self.df.columns[self.meta_cols] 

  #Define columns that can be analised
  def features(self):
    return self.df.columns[3:]

  #Define columns containing categorical infomration
  def cat_features(self):
    return ["SEX", "TT Response", "Measles Response"]  ### SI PUO MIGLIROARE 

  #Define columns containing date information
  def date_cols(self):
    return self.df.columns[self.dates]  ########SI PUO MIGLIORARE PER CAPIRE DA SOLO QUALI SIANO (IL DATABASE VA CONSIDERATO PULITO ORMAI)

  #Define columns related to clinical data
  def clinical_feat(self):
    return self.df.columns[4:11]

  #Define columns related to FACS data
  def facs_feat(self):
    return self.df.columns[41:371]

  def wb_abs(self):
    return ['p17', 'p24', 'p55', 'p31', 'p51', 'p66', 'gp41', 'gp120', 'gp160','p39']
  
  #Function for statistical analysis 
  #Given two groups of data and a list of feature of interest:
  #The appropriate test is chosen to test the hypothesis of different distributions
  ##If data are categoric a Figher's Exact test is performed
  #Else, normality assumption is tested via Shapiro Wilk's test 
  ##If True, homogeneity of variance is tested via Levene test 
  ###If True a two sided Welch ttest with equal variance is performed else with unequal assumption
  #If normality is not met:
  #Mann Whitney U test is performed
  #Results of the tests and the respective statistics are then collected inside a dictionary 
  def test_groups(self, 
                  g1, 
                  g2, 
                  it, 
                  p = 0.05, 
                  category = "Group", #Default use Group column
                  correction = "bonferroni", 
                  all = False): 
    de_mark = {}
    p_values = []
    test_names = []
    results_facs = []
    results_wb = []
    n = 0
    for i in it:
      if i in self.metadata():  
        #if i in self.df[i].dtype == 'object':
        if i in self.cat_features():
          test_name = "Fisher's exact test"
          obs_x = pd.crosstab(g1[i], 
                              g1[category]).reindex(index=g1[i].unique(), 
                                                    fill_value=0)
          print(obs_x)
          obs_y = pd.crosstab(g2[i], 
                              g2[category]).reindex(index=g2[i].unique(), 
                                                    fill_value=0)
          print(obs_y)
          try:
            odds, pval = stats.fisher_exact(np.array(obs_x, 
                                                     obs_y), 
                                            alternative='two-sided')
          except TypeError:
            try:
              ask_user = input(f"Which class should I compare?\n- Available: - One of {g2[category].dropna().unique()}\n- Merge (+)\n")
              if ask_user != "+":
                odds, pval = stats.fisher_exact(np.array([obs_x[obs_x.columns[0]], 
                                                          obs_y[ask_user]]), 
                                                alternative='two-sided')
              else:
                obs_y["+"] = obs_y[obs_y.columns[0]] + obs_y[obs_y.columns[1]]
                odds, pval = stats.fisher_exact(np.array([obs_x[obs_x.columns[0]], 
                                                          obs_y["+"]]), 
                                                alternative='two-sided')
            except (IndexError, ValueError) as e:
              print("ERROR")
              print(i)
              continue

          #all specifies whther to report all test results or only the significant
          if all:
            de_mark[i] = {"Test": test_name, 
                          "odds": odds, 
                          "p-value": np.format_float_scientific(pval, 2), 
                          "p-adj": np.format_float_scientific(pval, 2)}
          #reports only significant values
          else:
            if pval < p:
              de_mark[i] = {"Test": test_name, 
                            "odds": odds, 
                            "p-value": np.format_float_scientific(pval, 2), 
                            "p-adj": np.format_float_scientific(pval, 2)}
      elif pd.api.types.is_numeric_dtype(self.df[i]):
        try: 
          #validate normality assumption
          if norm_test(g1, g2, i, "SW"):
            #parametric test
            #validate equal variance assumption 
            if var_test(g1, g2, i):
              #welch ttest with equeal variance
              test_name = "T-test_ind, equal var"
              stat_test = stats.ttest_ind(g1[i],
                                            g2[i],
                                            equal_var = False,
                                            nan_policy = "omit",
                                            alternative = "two-sided")
            else:
              #welch ttest with unequeal variance
              test_name = "T-test_ind, unequal var"
              stat_test= stats.ttest_ind(g1[i],
                                            g2[i],
                                            equal_var = True,
                                            nan_policy = "omit",
                                            alternative = "two-sided")
          #non-parametric test
          else:
            #Mann Whitney U-test
            test_name = "Mann-Whitney U test"
            stat_test = stats.mannwhitneyu(g1[i].dropna(),
                                           g2[i].dropna(),
                                           alternative = "two-sided")
          
          #save results separately for abs and FACS features to perform bonferroni correction
          if i in self.facs_feat():
            results_facs.append([i, test_name, stat_test.pvalue])
          
          elif i in self.wb_abs():
            results_wb.append([i, test_name, stat_test.pvalue])

        except ValueError:
          continue

        #save statistics: number, means, sd, CI
        g1_n, g1_m, g1_sd, start1, end1 = confidence_interval(g1[i])
        g2_n, g2_m, g2_sd, start2, end2 = confidence_interval(g2[i])

        if g2_m != 0:
          fc = g1_m / g2_m 
        
        #adjustment for log2FC
        else:
          fc = 1
        if fc == 0:
          fc = 1 

        dict_result_no_filter(de_mark,
                                i,
                                test_name,
                                stat_test,
                                g1_n,
                                g2_n,
                                g1_m,
                                g2_m,
                                start1,
                                start2,
                                end1,
                                end2,
                                g1_sd,
                                g2_sd,
                                fc)

    if correction:
      results_facs = multiple_comparisons_correction(results_facs, correction)
      results_wb = multiple_comparisons_correction(results_wb, correction)
      res_corr = results_facs + results_wb
      for i in de_mark:
        if i in [sublist[0] for sublist in res_corr]:

          def get_value(value_to_match, list_of_lists):
            for element in list_of_lists:
              if element[0] == value_to_match:
                return float(element[2])

          de_mark[i]["p-adj"] = get_value(i, res_corr)
        else:
          de_mark[i]["p-adj"] = float(de_mark[i]["p-value"])

    if all:
      return de_mark
    else:
      de_mark = {k: v for k, v in de_mark.items() if v["p-adj"] < 0.05}
      for key in de_mark:
        inner_dict = de_mark[key]
        inner_dict["p-adj"] = np.format_float_scientific(inner_dict["p-adj"], 2)
      return de_mark
  
  def suppl_table_2(self, print_all = True):
    time_points = self.time_points
    de_final_res = {tp: {} for tp in time_points}

    for tp in de_final_res:
      de_final_res[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], self.hei_df[self.hei_df["Age"] == tp], self.features(), correction = "bonferroni", all = print_all)

    # Create a list to store the intermediate DataFrames
    tab_df_list = []

    # Convert the dictionary of dictionaries to a Pandas DataFrame
    for i in de_final_res:
      # Create a DataFrame from the inner dictionary
      inner_df = pd.DataFrame.from_dict(de_final_res[i], orient='index')
      # Add the current index to the DataFrame
      inner_df['index'] = i
      # Append the current DataFrame to the list of DataFrames
      tab_df_list.append(inner_df)  

    # Concatenate the list of DataFrames into one
    table_df = pd.concat(tab_df_list)

    # Set the index as a multi-index with 'index' and the original index
    table_df.set_index(['index', table_df.index], inplace=True)
  
    return table_df

  def make_pca_df(self, save = False, outFile = None):
    pca_df = self.df[['STUDY ID', 
                      'AGE IN MONTHS (days/30)', 
                      'Group', 
                      "VISIT N", 
                      "AGE IN MONTHS", 
                      "AGE IN DAYS"] + list(list(self.facs_feat())) + ["Measles Serology", 
                                                                       "Tetanous Serology", 
                                                                       "Measles Response", 
                                                                       "TT Response", 
                                                                       "WB score", 
                                                                       "EXTRA VACCINATION", 
                                                                       "AGE of ART initiation in days", 
                                                                       "Entry Viremia", 
                                                                       "HIV cp/mL", 
                                                                       "CD4%", 
                                                                       "WHO stage", 
                                                                       "Age"]]

    pca_df = pca_df[pca_df[self.facs_feat()].isnull().sum(axis = 1) <= (len(self.facs_feat())-1)]

    #control that less than 0.1% od entries are NA
    if pca_df.isna().sum().sum() / (pca_df.shape[0] * pca_df.shape[1]) >= 0.1:
      print("ERROR")

    #Assigning values measured at 9 months as 10 months, if no 10 months measurement
    for ind, row in pca_df.iterrows():
      if row["Age"] == 10:
        id = row["STUDY ID"]
        if type(row["TT Response"]) == float:
          pca_df.at[ind, "TT Response"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 9)]["TT Response"].values[0]
          pca_df.at[ind, "Tetanous Serology"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 9)]["Tetanous Serology"].values[0]

    #Assigning values measured at 2 months as entry, if no entry measurement
    for ind, row in pca_df.iterrows():
      if row["Age"] == "entry":
        id = row["STUDY ID"]
        if type(row["Measles Response"]) == float:
          if 2 in self.df[self.df["STUDY ID"] == id]["Age"].unique():
            pca_df.at[ind, "Measles Response"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 2)]["Measles Response"].values[0]
            pca_df.at[ind, "Measles Serology"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 2)]["Measles Serology"].values[0]

    pca_df = pca_df[pca_df["Age"].isin(["entry", 5., 10., 18., 19.])]
    pca_df.drop(labels = "Age", axis = 1)
    
    if save and outFile is not None:
        with pd.ExcelWriter(f"{outFile}",
                            datetime_format = 'YYYY-MM-DD',
                            ) as writer:
            pca_df.to_excel(writer, "Sheet1", index = False)
            
    return pca_df

  def make_ml_dataset(self):
    #retaining only datas with nonmissing values in FACS data
    ml_df = self.df[self.df[self.facs_feat()].isnull().sum(axis = 1) <= (len(self.facs_feat())-1)]

    #Assigning values measured at 9 months as 10 months, if no 10 months measurement
    for ind, row in ml_df.iterrows():
      if row["Age"] == 10:
        id = row["STUDY ID"]
        if type(row["TT Response"]) == float:
          ml_df.at[ind, "TT Response"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 9)]["TT Response"].values[0]
          ml_df.at[ind, "Tetanous Serology"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 9)]["Tetanous Serology"].values[0]

    #Assigning values measured at 2 months as entry, if no entry measurement
    for ind, row in ml_df.iterrows():
      if row["Age"] == "entry":
        id = row["STUDY ID"]
        if type(row["Measles Response"]) == float:
          if 2 in self.df[self.df["STUDY ID"] == id]["Age"].unique():
            ml_df.at[ind, "Measles Response"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 2)]["Measles Response"].values[0]
            ml_df.at[ind, "Measles Serology"] = self.df[(self.df["STUDY ID"] == id) & (self.df["Age"] == 2)]["Measles Serology"].values[0]

    #selecting columns for machine learning
    ml_df = ml_df[["STUDY ID", "AGE IN DAYS", "SEX", "WEIGHT", "HEIGHT", "Group", "FEEDING PRACTICE"] + 
                [col for col in self.facs_feat()] + ["HIV cp/mL"] + ["TT Response"] + ["Age"]]    
    
    #changing HEU NA values with 0
    ml_df["HIV cp/mL"] = np.where(ml_df["Group"] == "HEU", 0, ml_df["HIV cp/mL"])

    #Retaining only HEI entry data  
    ml_df_hei = ml_df[(ml_df["Group"] == "HEI") & (ml_df["Age"] == "entry")].drop(columns = ["Group"], inplace = False)
    ml_df_hei.reset_index(drop=True, inplace=True)

    #remove columns with more than 20% nas
    perc = 20.0 # Like N %
    ml_df_hei_clean = ml_df_hei.dropna(axis=1, thresh = int(((100-perc)/100)*ml_df_hei.shape[0] + 1))
    ml_df_hei_clean.reset_index(drop=True, inplace=True)
    
    #Assign values of TT response at 10 months to entry else na
    for ind, row in ml_df_hei_clean.iterrows():
      try:
        ml_df_hei_clean.at[ind, "TT Response"] = ml_df[(ml_df["STUDY ID"] == row["STUDY ID"]) & (ml_df["Age"] == 10.)]["TT Response"].values[0]
      except IndexError:
        ml_df_hei_clean.at[ind, "TT Response"] = np.nan

    #remove rows with na values
    ml_df_hei_clean.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)
    ml_df_hei_clean.reset_index(drop = True, inplace = True)
    #remove confounding columns
    ml_df_hei_clean.drop(columns = ["STUDY ID",
                                    "Age"], inplace = True)
    #binarize TT Response variable and set it to category
    ml_df_hei_clean["TT Response"] = np.where(ml_df_hei_clean["TT Response"] == "P", 1, 0)
    ml_df_hei_clean["TT Response"] = ml_df_hei_clean["TT Response"].astype("category")

    #preprocessing tge DataFrame by separating the numerical and non-numerical (categorical) columns, 
    #and converting the categorical columns to numerical format.
    ml_df_hei_clean_num = ml_df_hei_clean.select_dtypes(include=[np.number])
    ml_df_hei_clean_nonum = ml_df_hei_clean[pd.Index(set(ml_df_hei_clean.columns).difference(set(ml_df_hei_clean.select_dtypes(include=(np.number)).columns)))]
    #set categorical columns as category
    for col in ml_df_hei_clean_nonum.columns:
      ml_df_hei_clean_nonum[col] = ml_df_hei_clean_nonum[col].astype("category")
      ml_df_hei_clean_nonum[col] = ml_df_hei_clean_nonum[col].cat.codes
    
    #merge and return the final df
    ml_df_hei_clean = pd.concat([ml_df_hei_clean_num, ml_df_hei_clean_nonum], axis = 1)
    return(ml_df_hei_clean)

  def test_ml_algorithms(self, save = False, outFile = None):
    '''function to make an exploration of performance betweeh different ML algorithms'''
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split
    import plotly.figure_factory as ff

    df = self.make_ml_dataset()
    X = df.drop(columns = ["TT Response"], inplace = False)
    y = df["TT Response"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state =42)

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    models2 = models.insert(0, column = "Model", value= models.index.values)
    #fig = ff.create_table(models.round(decimals = 2))
    #fig.update_layout(autosize=True)
    #fig.write_image("table_plotly.png", scale=5)
    print(models)
    plt.figure(figsize = (10,10))
    idx = [i for i in range(1,28)]
    plt.plot(idx, models["Accuracy"]  ,marker='o' , label = "Accuracy", color = "limegreen" )
    plt.plot(idx , models["ROC AUC"] , marker ='o' , label = "ROC AUC", color = "magenta")
    plt.plot(idx , models["F1 Score"] , marker ='o' , label = "F1 Score", color = "red")

    plt.annotate("QDA", 
                  (1, models["Accuracy"][0]) , 
                  xytext  =(1, models["Accuracy"][0] - 0.2),
                  arrowprops = dict(
                                    arrowstyle = "simple"
                                   ),
                 fontsize = 12)

    plt.annotate(models.index[2], 
                  (3, models["Accuracy"][2]) , 
                  xytext  =(4, models["Accuracy"][2]+ 0.05),
                  arrowprops = dict(
                                    arrowstyle = "simple"
                                   ),
                 fontsize = 15,
                 color = "blue")
    plt.plot(3, models["Accuracy"][2], 'o', markersize = 10, mec='black', mfc='none')

    plt.xlabel("Models", size = 20)
    plt.ylabel("Metrics", size = 20)
    plt.title("Comparison of 27 Different Classifiers", size = 25, y = 1, pad = 40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.legend(prop={'size': 20}, markerscale=1., title_fontsize = "xx-large", fancybox = True, loc = "upper right")
  
    if save and outFile is not None:
      plt.savefig(outFile, dpi=600)

    plt.show()

  def train_nested_xgb(self, outFile = 'model_xgb'):
    '''outFile is path + outName (ex: path + "model_xgb"'''
    from hyperopt import fmin, tpe, hp, anneal, Trials
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    from sklearn.metrics import f1_score, accuracy_score
    from numpy import mean
    from numpy import arange
    from numpy import std
    from numpy import absolute
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.feature_selection import RFECV
    from xgboost import XGBClassifier


    df = self.make_ml_dataset()
    X = df.drop(columns = ["TT Response"], inplace = False)
    y = df["TT Response"]

    # possible values of parameters that the optimization algorithm can select from during the inner loop.
    space={'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
           'max_depth' : hp.quniform('max_depth', 2, 20, 1),
           'min_child_weight' : hp.quniform('min_child_weight', 1, 4, 1),
           'learning_rate': hp.loguniform('learning_rate', -5, 0),
           'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
           'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
           'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
          }
    #list to save results to seam average score
    outer_results = list()
    #initialize LOO crossvalidation
    cv_outer = LeaveOneOut()
    # initialize variables to store best hyperparameters and features
    best_hyperparameters = {}
    best_features = []
    best_score = float('inf')
    i = 0
    #perform cross validation for model stability/accuracy
    for train_ix, test_ix in cv_outer.split(X):
      X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
      y_train, y_test = y[train_ix], y[test_ix]

      #define minimization function for fmin: hyperparameter optimization
      def gb_f1_cv(params, random_state=32, cv=LeaveOneOut(), X=X_train, y=y_train):
        '''the function gets a set of variable parameters in "param"'''
        #params specify the hyperparameters for the XGBoost model for each iteration of the outer loop
        params = {'n_estimators': int(params['n_estimators']),
                  'max_depth': int(params['max_depth']),
                  'learning_rate': params['learning_rate']}

        # we use this params to create a new LGBM Regressor
        model = XGBClassifier(random_state=random_state, **params)

        # and then conduct the cross validation with the same folds as before
        #minimizing the negative of the accuracy score, which is equivalent to maximizing the accuracy score
        score = -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

        return score
        
      trials = Trials()
      #find optimized hyperparameters based on inner crossvalidation
      best=fmin(fn=gb_f1_cv, # function to optimize
                space=space,
                algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=15, # maximum number of iterations
                trials=trials, # logging
                rstate=np.random.RandomState(42) # fixing random state for the reproducibility
              )

      # computing the score on the test set
      #initializing XGB classificator with optimized hyperparameter based on inner-CV
      xgb_model = XGBClassifier(random_state=42, n_estimators=int(best['n_estimators']),
                            max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
      # Print the best F1 score and its corresponding hyperparameters
      print("Best f1 {:.3f} with hyperparameters {}".format(gb_f1_cv(best), best))
      # Print the best F1 score and its corresponding hyperparameters
      score = gb_f1_cv(best)
      #recursive feature elimination with automatic tuning of the number of features selected with cross-validation.
      rfecv_model = RFECV(
        estimator=xgb_model,
        step=1,
        cv=LeaveOneOut(),
        scoring="accuracy",
        min_features_to_select=1,
      )
      #fit the model
      rfecv_model.fit(X_train,y_train)
      # select the best features from the rfecv model
      selected_features = X_train.columns[rfecv_model.get_support()]
      print(f"RFECV selcted features: {selected_features}")
      # fit model with all data with best features from RFECV
      xgb_model.fit(X_train[selected_features], y_train)

      # Make predictions on the test set and append the accuracy score to a list
      y_pred = xgb_model.predict(X_test[selected_features])
      tpe_test_score= accuracy_score(y_test, y_pred)
      
      outer_results.append(tpe_test_score)
      print('> Accuracy=%.3f, F1 score=%.3f, Hyperparameters=%s' % (tpe_test_score, -gb_f1_cv(best), best))
      if score < best_score:
        if tpe_test_score == 1:
          best_score = score
          best_hyperparameters = best
          best_features = selected_features
          print("BEST")
          print(best_features)
      #saving intermidiate models
      xgb_model.save_model(f"{outFile}_{i}.model")
      with open(f"{outFile}_{i}.sav", 'wb') as f:
        pickle.dump(xgb_model, f)
      i += 1
    #print result of the outer-CV
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print(outer_results)
    # train final model on all data with best hyperparameters and features
    final_model = XGBClassifier(random_state=42, n_estimators=int(best_hyperparameters['n_estimators']),
                        max_depth=int(best_hyperparameters['max_depth']), learning_rate=best_hyperparameters['learning_rate'])
    final_model.fit(X[best_features], y)
    
    # Save the trained XGBoost model to a file
    final_model.save_model(f"{outFile}_final.model")
    with open(f"{outFile}_final.sav", 'wb') as f:
      pickle.dump(final_model, f)
            
    # Generate SHAP feature importance plots
    self.plot_model_feat(model_path=outFile, X=X[best_features], y = y, save=True, outName=outFile)
    print(f"{best_features},\n{best_score},\n{best_hyperparameters}")

  def plot_model_feat(self, model_path, X=None, y = None, save=False, outName=None):
    '''outName is path + outName (ex: path + "model_xgb"'''
    import shap
    import pickle
    
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
    #use shap explainer to plot feature importances
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Visualize the first prediction's explanation
    shap_wf = shap.plots.waterfall(shap_values[0])
    plt.tight_layout()
    plt.savefig(f"{outName}_waterfall.png", dpi=600)
    if save and outName is not None:
      plt.savefig(f"{outName}_waterfall.png", dpi=600)
    
    fig_bar = shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"{outName}_barplot.png", dpi=600)
    if save and outName is not None:
      plt.savefig(f"{outName}_barplot.png", dpi=600)

    y_pred = model.predict_proba(X)
    #use binary performances function to plot result of the model on a dataset
    binary_performances(y, y_pred, outName=f"{outName}_bin_perf.png")

  def figure_1a(self):
    '''figure 1a represents the results of the Tetanous Serology distributions betwenn HEI, HEU and CS at each time point'''
    serology = "Tetanous Serology"
    time_points = ["entry", 2., 5., 9., 10., 18.]
    #test group funtion to perform statistical analysis of the distributions
    de_final_res = {tp: self.test_groups(self.heu_df[self.heu_df["Age"] == tp], 
                                         self.hei_df[self.hei_df["Age"] == tp], 
                                         [serology], all=True) 
                    for tp in time_points}
  
    tt_cs_df = self.cs_df[~self.cs_df[serology].isna()]

    #Experiments were conducted only at 2 month, for the analysis we are considering age 2 as the entry for CS group
    tt_cs_df.replace({'Age' : {2: "entry"}}, inplace = True)

    cs_heu_de_final_res = {tp: self.test_groups(tt_cs_df[tt_cs_df["Age"] == tp], 
                                                 self.heu_df[self.heu_df["Age"] == tp], 
                                                 [serology], "SW", all=True) 
                           for tp in time_points}
    
    cs_hei_de_final_res = {tp: self.test_groups(tt_cs_df[tt_cs_df["Age"] == tp], 
                                                 self.hei_df[self.hei_df["Age"] == tp], 
                                                 [serology], "SW", all=True) 
                           for tp in time_points}

    #Plotting code
    base_df = self.df[self.df["Age"].isin(time_points)]
    base_df["utils"] = base_df[["Group", "EXTRA VACCINATION"]].apply(tuple, axis = 1)
    base_df = base_df[base_df["Tetanous Serology"].notna()].sort_values(by = ["STUDY ID"])
    base_df.replace({"entry": 1., 2.: 1.}, inplace = True)

    #pairs and pvalues for statannotation
    pairs = []
    for b in np.sort(base_df["Age"].unique()):
        if b == 1.:
            b = "entry"
        if "Tetanous Serology" in de_final_res[b]:
            res_t = tuple((b, i) for i in ["HEU", "HEI"])
            pairs.append((res_t[0], res_t[1]))

    pvalues = []
    for age in np.sort(base_df["Age"].unique()):
      if age == 1.:
        age = "entry"
      if "Tetanous Serology" in de_final_res[age].keys():
        pvalues.append(float(de_final_res[age]["Tetanous Serology"]["p-value"]))

    pairs_heu = []
    for b in np.sort(base_df["Age"].unique()):
        if b == 1.:
            b = "entry"
        if "Tetanous Serology" in cs_heu_de_final_res[b]:
            res_heu = tuple((b, i) for i in ["HEU", "CS"])
            pairs_heu.append((res_heu[0], res_heu[1]))

    pvalues_heu = []
    for age in np.sort(base_df["Age"].unique()):
      if age == 1.:
        age = "entry"
      if "Tetanous Serology" in cs_heu_de_final_res[age].keys():
        pvalues_heu.append(float(cs_heu_de_final_res[age]["Tetanous Serology"]["p-value"]))

    pairs_hei = []
    for b in np.sort(base_df["Age"].unique()):
        if b == 1.:
            b = "entry"
        if "Tetanous Serology" in cs_hei_de_final_res[b]:
            res_hei = tuple((b, i) for i in ["HEI", "CS"])
            pairs_hei.append((res_hei[0], res_hei[1]))

    pvalues_hei = []
    for age in np.sort(base_df["Age"].unique()):
      if age == 1.:
        age = "entry"
      if "Tetanous Serology" in cs_hei_de_final_res[age].keys():
        pvalues_hei.append(float(cs_hei_de_final_res[age]["Tetanous Serology"]["p-value"]))
    
    sns.set_style("white")
    sns.set_context("poster")

    #plotting code
    fig, ax = plt.subplots(sharex = True, figsize = (28, 15))

    sns.stripplot(data=base_df, x="Age", y="Tetanous Serology",
                  edgecolor="black", linewidth=2, hue="Group",
                  hue_order=["CS", "HEU", "HEI"], palette=["#008000", "#1C86EE", "#FF8C00"],
                  dodge=True, size=15, ax=ax)

    sns.boxplot(showmeans=True, meanline=True, meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False}, whiskerprops={'visible': False}, zorder=10,
                x="Age", y="Tetanous Serology", hue="Group", hue_order=["CS", "HEU", "HEI"],
                data=base_df, showfliers=False, showbox=False, showcaps=False, ax=ax)

    sns.despine(offset=10, trim=True)

    plt.axhline(y=0.1, color='r', linestyle='--')
    plt.axhline(y=0.5, color='r', linestyle='--')

    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, 0.25, "0.1", color="red", transform=trans, ha="right", va="center")
    ax.text(0, 0.501, "0.5", color="red", transform=trans, ha="right", va="center")
    ax.text(0.04, 0.65, "LOD", color="red", transform=trans, ha="right", va="center")

    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    ax.set_xlabel(ax.get_xlabel(), fontsize=40)
    ax.set_xticklabels(["1-2", "5", "9", "10", "18"])
    ax.set_ylabel(ax.get_ylabel(), fontsize=40)
    ax.set_ylim(top=6)

    pairs = [((1.0, 'HEU'), (1.0, 'HEI')), ((1.0, 'HEU'), (1.0, 'CS')), ((1.0, 'HEI'), (1.0, 'CS')),
             ((5.0, 'HEU'), (5.0, 'HEI')), ((10.0, 'HEU'), (10.0, 'HEI')), ((18.0, 'HEU'), (18.0, 'HEI'))]
    pvalues = ["p = 0.001", "p < 0.001", "p < 0.001", "p = 0.006", "p = 0.019", "p = 0.039"]

    annotator = Annotator(ax, pairs, data=base_df, x="Age", y="Tetanous Serology",
                          hue="Group", hue_order=["CS", "HEU", "HEI"], order=[1., 5., 9., 10., 18.], line_offset_to_box=0.2)
    annotator.set_custom_annotations(pvalues)
    annotator.configure(loc="inside", text_format="full", line_width=3, fontsize="large", show_test_name=False)
    annotator.annotate()

    h, l = ax.get_legend_handles_labels()
    plt.legend(h[3:],['Control', 'HEU', 'HEI'], prop={'size': 25}, markerscale=2., fancybox = True)
    plt.ylabel("Tetanus Ab mlU/mL", labelpad = 40)
    plt.xlabel("Age in months", labelpad = 40)
    plt.title("Tetanus Serology in TARA cohort", fontsize = 40, y = 1, pad = 150, fontname="Times New Roman Bold")


    # Create pivot table and add it below the scatter plot
    threshold = 0.5
    base_df['Above/Below'] = ['Above' if value >= threshold else "Borderline" if 0.1 < value < threshold else 'Below' for value in base_df['Tetanous Serology']]
    base_df_n = base_df.drop(base_df[(base_df["Group"] == "CS") & (base_df["Age"] != 1.)].index)
    pivot = base_df_n.pivot_table(index='Above/Below', columns=['Age', "Group"],
                      values='Tetanous Serology', aggfunc=['count'])

    pivot.columns = [f"{col[2]}" for col in pivot.columns]
    table = ax.table(cellText=pivot.values, colLabels=pivot.columns, rowLabels=pivot.index, loc='center', bbox=[0, -0.4, 0.9, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(0.00002, 0.00002)

    plt.tight_layout()
    plt.show()

  def figure_1b(self):
    #Testing differences between HEI and HEU in WB score
    time_points = ["entry", 2., 5., 9., 10., 18.]
    de_final_res = {tp: {} for tp in time_points}  
    for tp in de_final_res:
      de_final_res[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], self.hei_df[self.hei_df["Age"] == tp], ["WB score"], all = True)
    
    #Steps to find paired datas between entry and 10 months
    paired = self.df[self.df["Age"].isin(["entry", 10.])]

    list_common_id = []
    for val, val_df in paired.groupby("Age"):
      list_common_id.append(list(val_df["STUDY ID"].unique()))

    paired = paired[paired["STUDY ID"].isin(list(set.intersection(*map(set,list_common_id))))]
    #Testing differences between HEI and HEU in WB score on paired samples
    wb_final_res = {gp: {} for gp in ["HEU", "HEI"]}
    for gp in ["HEU", "HEI"]:
      wb_final_res[gp] = self.test_groups(paired[(paired["Age"] == "entry") & (paired["Group"] == gp)], paired[(paired["Age"] == 10.) & (paired["Group"] == gp)], ["WB score"], "SW")
  
    #plotting part
    base_df = self.df[self.df["Age"].isin(["entry", 10.])]
    base_df_g = base_df.set_index('Age', append=True).set_index("Group", append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")

    col = ["WB score"]

    #pvalues and pairs for statannotation
    pvalues = []
    pairs = []

    for b in base_df["Age"].unique():
      if de_final_res[b] != {}:
        res_b = ()
        for i in ["HEI", "HEU"]:
          res_b += (b, i)
        pairs.append((res_b))

    for pairings in [(base_df["Age"].unique()[0], "HEU", base_df["Age"].unique()[1], "HEU"), (base_df["Age"].unique()[0], "HEI", base_df["Age"].unique()[1], "HEI")]:
      pairs.append(pairings)

    pairs = list(map(lambda x: ((x[0], x[1]), (x[2], x[3])), pairs))
    for t in base_df["Age"].unique():
      for i in col:
        if i in de_final_res[t]: 
          pvalues.append(float(de_final_res[t][i]["p-adj"]))

    for x in wb_final_res:
      pvalues.append(float(wb_final_res[x]["WB score"]["p-adj"]))

    res_box = base_df_g.loc[base_df_g["feature"].isin(col)]
    res_box = res_box.astype({"value": float})

    fig = plt.figure(figsize = (21,21))

    sns.set_style("white")
    sns.set_context("poster")

    ax = sns.violinplot(x = res_box["Age"], y = res_box["value"], hue = res_box["Group"], hue_order = ["HEU", "HEI"], order = ["entry", 10.], inner = None, cut = 0, linewidth = 0)

    plt.setp(ax.collections, alpha=.6)

    ax = sns.swarmplot(x = res_box["Age"], y = res_box["value"], hue = res_box["Group"], hue_order = ["HEU", "HEI"], order = ["entry", 10.], dodge = True, s = 10, color = 'k')

    ax = sns.boxplot(showmeans=True,
                     meanline=True,
                     meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                     medianprops={'visible': False},
                     whiskerprops={'visible': False},
                     zorder=10,
                     x="Age",
                     y="value",
                     hue = "Group",
                     hue_order = ["HEU", "HEI"],
                     order = ["entry", 10.],
                     data=res_box,
                     showfliers=False,
                     showbox=False,
                     showcaps=False,
                     ax=ax)

    sns.despine(offset=10, trim=True)
    sns.color_palette("Paired")

    plt.xlabel("Age in months", fontsize = 55)
    plt.ylabel("WB score", fontsize = 55)
    ax.tick_params(axis='both', which='major', labelsize=55)
    ax.tick_params(axis='both', which='minor', labelsize=35)
    ax.set_xticklabels(['1-2', '10'])

    annotator = Annotator(ax, pairs, data = res_box, x = "Age", y = "value", hue = "Group", hue_order = ["HEU", "HEI"], order = ["entry", 10.])
    formatted_pvalues = ["ns", "p < 0.001", "p < 0.001", "p < 0.001"]

    annotator.set_custom_annotations(formatted_pvalues)
    #annotator.set_pvalues(pvalues)
    annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

    ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
    ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
    annotator.annotate()

    adjust_box_widths(fig, 0.9)
    h,l = ax.get_legend_handles_labels()
    plt.legend(h[:2],['HEU', 'HEI'], prop={'size': 30}, markerscale=2., title_fontsize = "xx-large", fancybox = True, loc = "upper right")

    plt.tight_layout()
    plt.show()

  def figure_1c(self):
    '''Plotting differences in HIV Abs levels between HEI and HEU sat entry based on results obrained with test_groups()'''
    categories = ["p17", "p55", "gp160", "p39", "p31", "p51", "gp120", "gp41", "p66", "p24"]
    #renaming categories for plotting purpose based on the result of the de analysis
    categories.append(categories[0])
    labels = ["p17 ***", "p55 ***", "gp160 ***", "p39 *", "p31", "p51", "gp120", "gp41", "p66", "p24", "p17 ***"]

    fig = go.Figure()

    heu_vals = [val for val in self.df[(self.df["Age"] == "entry") & (self.df["Group"] == "HEU")][categories].apply(unify_ab_values, axis = 0)]
    heu_vals.append(heu_vals[0])
    hei_vals = [val for val in self.df[(self.df["Age"] == "entry") & (self.df["Group"] == "HEI")][categories].apply(unify_ab_values, axis = 0)]
    hei_vals.append(hei_vals[0])

    fig.add_trace(go.Barpolar(
        r=heu_vals,
        theta=labels,
        marker_line_color="black",
        marker_line_width=2,
        opacity=0.8,
        name = "HEU",
        base = (0,0)
    ))

    fig.add_trace(go.Barpolar(
        r=hei_vals,
        base = (0,0),
        theta=labels,
        marker_line_color = "black",
        marker_line_width = 2,
        opacity = 0.6,
        name='HEI'
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 1], visible=True, tickfont_size= 16, tickcolor = "black", tickfont = dict(family = "Arial Black")),
            angularaxis = dict(showticklabels=True, tickfont_size=20, tickfont = dict(family = "Arial Black"))
        ), showlegend = True
    )

    fig.show()

  def figure_1d(self):
    #Plotting differences in HIV Abs levels between HEI and HEU at 10 months based on results obrained with test_groups()
    categories = ["p17", "p55", "gp160", "p31", "p39", "p51", "gp120", "gp41", "p66", "p24"]
    #renaming categories for plotting purpose based on the result of the de analysis
    categories.append(categories[0])
    labels = ["p17 *", "p55", "gp160 **", "p31", "p39", "p51", "gp120 ***", "gp41 **", "p66", "p24 ***", "p17 *"]


    fig = go.Figure()

    heu_vals = [val for val in self.df[(self.df["Age"] == 10) & (self.df["Group"] == "HEU")][categories].apply(unify_ab_values, axis = 0)]
    heu_vals.append(heu_vals[0])
    hei_vals = [val for val in self.df[(self.df["Age"] == 10) & (self.df["Group"] == "HEI")][categories].apply(unify_ab_values, axis = 0)]
    hei_vals.append(hei_vals[0])


    fig.add_trace(go.Barpolar(
        r=heu_vals,
        theta=labels,
        marker_line_color="black",
        marker_line_width=2,
        opacity=0.8,
        name = "HEU",
        base = (0,0)
    ))

    fig.add_trace(go.Barpolar(
        r=hei_vals,
        base = (0,0),
        theta=labels,
        marker_line_color = "black",
        marker_line_width = 2,
        opacity = 0.6,
        name='HEI'
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 1], visible=True, tickfont_size= 16, tickcolor = "black", tickfont = dict(family = "Arial Black")),
            angularaxis = dict(showticklabels=True, tickfont_size=20, tickfont = dict(family = "Arial Black"))
        ), showlegend = True
    )

    fig.show()

  def figure_1e(self):
    #Testing differences between HEI TTP and HEI TT UP in WB score
    hei_final_res_resp = {}
    feature = ["WB score"]
    time_points = ["entry", 2., 5., 9., 10., 18.]
    feats_resp = self.feats_resp
    hei_final_res_resp = {feat: {} for feat in feats_resp}
    for i in time_points:
      try:
        hei_final_res_resp[feats_resp[0]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[0]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[0]].isin(["UP"])], 
                                                                feature,  
                                                                category = feats_resp[0], 
                                                                all = False)
        hei_final_res_resp[feats_resp[1]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[1]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[1]].isin(["UP"])], 
                                                                feature,  
                                                                category = feats_resp[1], 
                                                                all = False)
      except ValueError:
        continue
    print(hei_final_res_resp)
    #steps to find paired data between entry and 10 months 
    feature = "TT Response"

    paired = self.hei_df[self.hei_df["Age"].isin(["entry", 10.])]

    list_common_id = []
    for val, val_df in paired.groupby("Age"):
      list_common_id.append(list(val_df["STUDY ID"].unique()))

    paired = paired[paired["STUDY ID"].isin(list(set.intersection(*map(set,list_common_id))))]
    #Testing differences between HEI TTP and HEI TT UP in WB score on paired samples
    wb_final_res = {gp: {} for gp in ["P", "UP"]}  
    for gp in ["P", "UP"]:
      wb_final_res[gp] = self.test_groups(paired[paired["Age"] == "entry"][paired[paired["Age"] == "entry"][feature] == gp], paired[paired["Age"] == 10.][paired[paired["Age"] == 10.][feature] == gp], ["WB score"], "SW", all = False)

    #Plotting part
    base_df = self.hei_df[self.hei_df["Age"].isin(["entry", 10.])]
    base_df_g = base_df.set_index('Age', append=True).set_index(feature, append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")

    col = ["WB score"]
    #saving pairings and pvalues for statannotation
    pvalues = []
    pairs = []

    for b in base_df["Age"].unique():
      if col[0] in hei_final_res_resp[feature][b]: 
        res_b = ()
        for i in ["UP", "P"]:
          res_b += (b, i)
        pairs.append((res_b))

    for pairings in [(base_df["Age"].unique()[0], "UP", base_df["Age"].unique()[1], "UP"), (base_df["Age"].unique()[0], "P", base_df["Age"].unique()[1], "P")]:
      pairs.append(pairings)

    pairs = list(map(lambda x: ((x[0], x[1]), (x[2], x[3])), pairs))
    for t in base_df["Age"].unique():
      for i in col:
        if i in hei_final_res_resp[feature][t]: 
          pvalues.append(float(hei_final_res_resp[feature][t][i]["p-adj"]))

    for cat in ["UP", "P"]:
      pvalues.append(float(wb_final_res[cat]["WB score"]["p-value"]))

    res_box = base_df_g.loc[base_df_g["feature"].isin(col)]

    fig = plt.figure(figsize = (21,21))

    sns.set_style("white")
    sns.set_context("poster")

    res_box = res_box.astype({"value": float})
    ax = sns.violinplot(x = res_box["Age"], y = res_box["value"], hue = res_box[feature], hue_order = ["UP", "P"], order = ["entry", 10.], inner = None, cut = 0, linewidth = 0, palette = ["#FFCBA4", "#FF6700"])

    plt.setp(ax.collections, alpha=.6)

    ax = sns.swarmplot(x = res_box["Age"], y = res_box["value"], hue = res_box[feature], hue_order = ["UP", "P"], order = ["entry", 10.], dodge = True, s = 10, color = 'k')

    ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Age",
                y="value",
                hue = feature,
                hue_order = ["UP", "P"],
                order = ["entry", 10.],
                data=res_box,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

    sns.despine(offset=10, trim=True)
    sns.color_palette("Paired")

    plt.xlabel("Age in months", fontsize = 55)
    plt.ylabel("WB score", fontsize = 55)
    ax.tick_params(axis='both', which='major', labelsize=55)
    ax.tick_params(axis='both', which='minor', labelsize=35)
    ax.set_xticklabels(['1-2', '10'])


    annotator = Annotator(ax, pairs, data = res_box, x = "Age", y = "value", hue = feature, hue_order = ["UP", "P"], order = ["entry", 10.])
    formatted_pvalues = ["p < 0.001", "p = 0.003"]

    annotator.set_custom_annotations(formatted_pvalues)
    #annotator.set_pvalues(pvalues)

    annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

    ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
    ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
    annotator.annotate()

    adjust_box_widths(fig, 0.9)
    h,l = ax.get_legend_handles_labels()
    plt.legend(h[:2],['HEI TT UP', 'HEI TT P'], prop={'size': 30}, markerscale=2., title_fontsize = "xx-large", fancybox = True, loc = "best")
    fig.show()

  def figure_1f(self):
    #Plotting differences in HIV Abs levels between HEI TT P and HEI TT UP at entry based on results obtained with test_groups()
    feature = "TT Response"
    categories = ["p17", "p55", "gp160", "p39", "p31", "p51", "gp120", "gp41", "p66", "p24"]

    base_df = self.hei_df[self.hei_df["Age"].isin(["entry", 10.])]

    heu_vals = [val for val in base_df[(base_df["Age"] == "entry") & (base_df[feature] == "UP")][categories].apply(unify_ab_values, axis = 0)]
    heu_vals.append(heu_vals[0])
    hei_vals = [val for val in base_df[(base_df["Age"] == "entry") & (base_df[feature] == "P")][categories].apply(unify_ab_values, axis = 0)]
    hei_vals.append(hei_vals[0])
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=heu_vals,
        theta=categories,
        marker_line_color="black",
        marker_color = "#FFCBA4",
        marker_line_width=2,
        opacity=0.6,
        name = "HEI UP",
        base = (0,0)
    ))

    fig.add_trace(go.Barpolar(
        r=hei_vals,
        base = (0,0),
        theta=categories,
        marker_line_color = "black",
        marker_color = "#FF6700",
        marker_line_width = 2,
        opacity = 0.6,
        name='HEI P'
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 1], visible=True, tickfont_size= 16, tickcolor = "black", tickfont = dict(family = "Arial Black")),
            angularaxis = dict(showticklabels=True, tickfont_size=20, tickfont = dict(family = "Arial Black"))
        ), showlegend = True
    )

    fig.show()

  def figure_1g(self):
    #Plotting differences in HIV Abs levels between HEI TT P and HEI TT UP at 10 months based on results obtained with test_groups()
    categories = ["p17", "p55", "gp160", "p39", "p31", "p51", "gp120", "gp41", "p66", "p24"]
    #renaming categories for plotting purpose based on the result of the de analysis
    categories.append(categories[0])
    labels = ["p17", "p55", "gp160", "p39", "p31", "p51", "gp120", "gp41", "p66 *", "p24", "p17"]

    feature = "TT Response"

    base_df = self.hei_df[self.hei_df["Age"].isin(["entry", 10.])]
    base_df.replace({feature: {"R": "P", "LTM": "P", "RB": "UP", "NM": "UP", "PB": "UP", "MB": "UP", "UR": "UP"}}, inplace = True)

    heu_vals = [val for val in base_df[(base_df["Age"] == 10.) & (base_df[feature] == "UP")][categories].apply(unify_ab_values, axis = 0)]
    heu_vals.append(heu_vals[0])
    hei_vals = [val for val in base_df[(base_df["Age"] == 10.) & (base_df[feature] == "P")][categories].apply(unify_ab_values, axis = 0)]
    hei_vals.append(hei_vals[0])

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=heu_vals,
        theta=labels,
        marker_line_color="black",
        marker_color = "#FFCBA4",
        marker_line_width=2,
        opacity=0.6,
        name = "HEI UP",
        base = (0,0)
    ))

    fig.add_trace(go.Barpolar(
          r=hei_vals,
          base = (0,0),
          theta=labels,
          marker_line_color = "black",
          marker_color = "#FF6700",
          marker_line_width = 2,
          opacity = 0.6,
          name='HEI P'
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 1], visible=True, tickfont_size= 16, tickcolor = "black", tickfont = dict(family = "Arial Black")),
            angularaxis = dict(showticklabels=True, tickfont_size=20, tickfont = dict(family = "Arial Black"))
        ), showlegend = True
    )

    fig.show()

  def figure_2a(self):
    #Plotting differences in HIV cp/mL levels between HEI and HEU at every tp based on results obtained with test_groups()
    hei_final_res_resp = {}
    col = ["HIV cp/mL"]   
    hei_final_res_resp = {feat: {} for feat in self.feats_resp}
    for i in self.time_points:      
      try:
        hei_final_res_resp[self.feats_resp[0]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][self.feats_resp[0]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][self.feats_resp[0]].isin(["UP"])], 
                                                                col, 
                                                                category = self.feats_resp[0], 
                                                                all = False)
        hei_final_res_resp[self.feats_resp[1]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][self.feats_resp[1]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][self.feats_resp[1]].isin(["UP"])], 
                                                                col,  
                                                                category = self.feats_resp[1], 
                                                                all = False)
      except ValueError:
        continue
    print(hei_final_res_resp)
    #Plotting part
    feature = "TT Response"
    base_df = self.hei_df[(~self.hei_df["Tetanous Serology"].isna()) & (~self.hei_df["HIV cp/mL"].isna())]
    base_df_g = base_df.set_index('Age', append=True).set_index(feature, append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")
    base_df_g = base_df_g[base_df_g["Age"].isin(["entry", 5., 9., 10., 18.])]

    #pairs and pvalues for statannotation
    pairs = []
    pvalues = []
    for t, x in base_df_g.groupby("Age"):
      if t in hei_final_res_resp[feature].keys():
        if col[0] in hei_final_res_resp[feature][t]:
          pvalues.append(float(hei_final_res_resp[feature][t][col[0]]["p-adj"]))
          res_l = ()
          for clas in ["P", "UP"]:
            res_l += (t, clas)
          pairs.append((res_l))
    pairs = [((par[0], par[1]), (par[0], par[-1])) for par in pairs]
    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]

    res_box = base_df_g.loc[base_df_g["feature"] == col[0]]
    res_box = res_box.astype({"value": float})

    fig = plt.figure(figsize = (30,15))

    sns.set_style("white")
    sns.set_context("poster")
    ax = sns.violinplot(x = res_box["Age"], y = res_box["value"], 
                        hue = res_box["TT Response"], hue_order = ["UP", "P"], 
                        inner = "point", cut = 0, scale = "width", 
                        linewidth = 0, palette = ["#FFCBA4", "#FF6700"])

    plt.setp(ax.collections, alpha=.6)
    ax = sns.stripplot(x = res_box["Age"], y = res_box["value"], 
                       hue = res_box["TT Response"], hue_order = ["UP", "P"], 
                       dodge = True, s = 7, color = 'k')

    ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Age",
                y="value",
                hue = "TT Response",
                hue_order = ["UP", "P"],
                data=res_box,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

    sns.despine(offset=10, trim=True)

    ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
    ax.set_xticklabels(["1-2", 5, 9, 10, 18])
    ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
    ax.yaxis.offsetText.set_fontsize(30)

    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=30)

    #adjust_box_widths(fig, 2)

    plt.xlabel("Age in months", fontsize = 40)
    plt.ylabel("HIV cp/mL", fontsize = 40)

    annotator = Annotator(ax, pairs, data = res_box, x = "Age", y = "value", hue = feature, hue_order = ["UP", "P"])
    formatted_pvalues = ["p = 0.02", "p = 0.005", "p = 0.02"]

    annotator.set_custom_annotations(formatted_pvalues)

    #annotator.set_pvalues(pvalues)
    annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

    annotator.annotate()

    h,l = ax.get_legend_handles_labels()
    plt.legend(h[:2],['HEI TT UP', 'HEI TT P'], prop={'size': 25}, markerscale=2., title_fontsize = "xx-large", fancybox = True)
    plt.title(f"HIV cp/mL in Tetanus Protected and Unprotected HEI patients", fontsize = 40, y = 1, pad = 100, fontname="Times New Roman Bold")

    plt.tight_layout()
    plt.show()

  def figure_2b(self):
    #Plotting differences in mean AUC HIV cp/mL levels between HEI TT P and HEI TT UP at every time point based on results obtained with test_groups()
    plt.style.use('default')

    fig, ax = plt.subplots(1, 1, figsize=(20,20))
    #Selecting HEI group and data corresponding to time points chosen
    data = self.hei_df[~self.hei_df["TT Response"].isna()]
    data = data[data["Age"].isin(["entry", 9., 10., 18.])]
    
    #computing statistics based on Protected patients: mean
    mean_p = data[data["TT Response"].isin(["P", "R", "LTM"])].groupby("Age")["HIV cp/mL"].mean().tolist()
    age_p = data[data["TT Response"].isin(["P", "R", "LTM"])].groupby("Age").mean().index.tolist()

    #computing statistics based on Unprotected patients: mean
    mean_p_ur = data[~data["TT Response"].isin(["P", "R", "LTM"])].groupby("Age")["HIV cp/mL"].mean().tolist()
    age_p_ur = data[~data["TT Response"].isin(["P", "R", "LTM"])].groupby("Age").mean().index.tolist()

    #changing names for plotting purporse
    res = pd.DataFrame({"Age in months": age_p, "HIV cp/mL": mean_p}).replace("entry", 1.0).sort_values("Age in months")
    res_ur = pd.DataFrame({"Age in months": age_p_ur, "HIV cp/mL": mean_p_ur}).replace({"Age in months": {2.0: 1.0}}).replace("entry", 1.0).sort_values("Age in months")

    #computing AUC
    auc_p = sklearn.metrics.auc(res["Age in months"], res["HIV cp/mL"])
    auc_up = sklearn.metrics.auc(res_ur["Age in months"], res_ur["HIV cp/mL"])
    #plots
    res_ur.plot.area(x="Age in months", y="HIV cp/mL", label=f"UP: log(AUC) = {round(np.log10(auc_up), 2)}", stacked=False, ax=ax, linewidth=4, linestyle="--", alpha=0.2, color="#FFCBA4", ylabel="HIV cp/mL", title="HEI Mean AUC HIV cp/mL")
    res.plot.area(x="Age in months", y="HIV cp/mL", label=f"P: log(AUC) = {round(np.log10(auc_p), 2)}", stacked=False, ax=ax, linewidth=4, linestyle="--", alpha=0.2, color="#FF6700", ylabel="HIV cp/mL", title="Mean AUC of HIV cp/mL in HEI")

    p_df = data[data["TT Response"].isin(["P", "R", "LTM"])].replace("entry", 1.0)
    sns.lineplot(data=p_df, x="Age", y="HIV cp/mL", color="#FF6700", alpha=0.95, linewidth=2, linestyle="--")

    up_df = data[~data["TT Response"].isin(["P", "R", "LTM"])].replace("entry", 1.0)
    sns.lineplot(data=up_df, x="Age", y="HIV cp/mL", color="#FFCBA4", alpha=0.95, linewidth=2, linestyle="--")

    ax.set_ylim(20, 15000000)
    ax.set_yscale('log')

    for axis in [ax.xaxis, ax.yaxis]:
      axis.set_tick_params(labelsize=45)

    ax.set_xlabel("Age in months", fontsize=50, labelpad=40)
    ax.set_ylabel(ax.get_ylabel(), fontsize=50, labelpad=40)

    ax.yaxis.offsetText.set_fontsize(30)

    plt.scatter(res["Age in months"], res["HIV cp/mL"], color = 'k')
    plt.scatter(res_ur["Age in months"], res_ur["HIV cp/mL"], color = 'k')

    plt.text(9.7, 0.13 * 10000000, "*", fontsize = 50) #mark significance at 10 months as of de_final_res

    plt.xticks(ticks = [1, 9, 10, 18], labels = ["1-2", 9, 10, 18])
    plt.legend(prop={'size': 25}, markerscale=3., title = "Tetanus Response", title_fontsize = 30)
    ax.set_title(ax.get_title(), fontsize = 55, y = 1.1)

    plt.tight_layout()

  def figure_3a(self):
    '''Plotting results of DE in FACS features between HEI and HEU at every time points based on results obtained with test_groups() as barplot'''
    time_points = ["entry", 5., 9., 10., 18.]
    facs_de = {tp: {} for tp in time_points}
    
    #DE analysis
    for tp in facs_de:
      facs_de[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], 
                                     self.hei_df[self.hei_df["Age"] == tp], 
                                     self.facs_feat(), 
                                     correction = "bonferroni", 
                                     all = False)
    #Barplot
    bot = [len(facs_de["entry"]), len(facs_de[5.]), len(facs_de[9.]), len(facs_de[10.]), len(facs_de[18.])]
    heig = [len(self.facs_feat()), len(self.facs_feat()), len(self.facs_feat()), len(self.facs_feat()), len(self.facs_feat())]
    fig, ax = plt.subplots(1, 1, figsize = (15,15))
    plt.style.use("default")
    plt.bar([0, 1, 2, 3, 4], height = bot, tick_label = ["1-2", 5, 9, 10, 18], color = "green", edgecolor = "k", linewidth = 2.5)
    plt.title("DE features per time point", fontsize = 40, y = 1.1)
    plt.xlabel('Age in months')
    plt.ylabel('FACS features')
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 40, labelpad = 40)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 40, labelpad = 40)
    plt.tight_layout()
    plt.show()

  def figure_3b(self):
    #Plotting results of DE in FACS features between HEI and HEU at entry based on results obtained with test_groups()
    base_df = self.df
    base_df_g = base_df.set_index('Age', append=True).set_index("Group", append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")
    time_points = ["entry", 2., 5., 9., 10., 18.]
    #DE analysis
    de_final_res = {tp: {} for tp in time_points}
    for tp in de_final_res:
      de_final_res[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], self.hei_df[self.hei_df["Age"] == tp], self.facs_feat(), all = True)
    
    #Plotting code
    for t, x in base_df_g.groupby("Age"):
      #selection of features based on DE analysis
      col = ["Transitional %", "Mature B cells %", "Naive %", "DN %", "DN/CD21- %"]
      #Selection of time point
      if t == "entry":
        #pvalues and pairs for statannotation
        pvalues = []
        pairs = []

        for b in col:
          res_b = ()
          for i in ["HEU", "HEI"]:
            res_b += (b, i)
          pairs.append((res_b))
        pairs = [((i[0], i[1]), (i[0], i[-1])) for i in pairs]

        for i in col:
          pvalues.append(float(de_final_res[t][i]["p-adj"])) #cambiato pvalue con p-adj

        res_box = x.loc[x["feature"].isin(col)]
        fig = plt.figure(figsize = (45, 15)) 

        sns.set_style("white")
        sns.set_context("poster")


        if col == []:
          continue
        res_box = res_box.astype({"value": float})

        ax = sns.violinplot(x = res_box["feature"],
                          y = res_box["value"],
                          hue = res_box["Group"],
                          hue_order = ["HEU", "HEI"],
                          order = col,
                          inner = None,
                          cut = 0,
                          linewidth = 0)

        plt.setp(ax.collections, alpha=.6)

        ax = sns.swarmplot(x = res_box["feature"],
                         y = res_box["value"],
                         hue = res_box["Group"],
                         hue_order = ["HEU", "HEI"],
                         order = col,
                         dodge = True,
                         s = 10,
                         color = 'k')

        ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="feature",
                y="value",
                hue = "Group",
                hue_order = ["HEU", "HEI"],
                order = col,
                data=res_box,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

        sns.despine(offset=10, trim=True)
        sns.color_palette("Paired")

        plt.xlabel("")
        plt.ylabel("% of CD19+CD20+ B cells", fontsize = 50)
  
        ax.tick_params(axis='both', which='major', labelsize=55)

        annotator = Annotator(ax, pairs, data = res_box, x = "feature", y = "value", hue = "Group", hue_order = ["HEU", "HEI"], order = col)

        #Formatting p values for plotting purpose
        formatted_pvalues = ["p = 0.005", "p = 0.003", "p = 0.006", "p < 0.001", "p < 0.001"]

        annotator.set_custom_annotations(formatted_pvalues)

        annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

        ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
        ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
        annotator.annotate()

        labels = ["CD10+\nTransitional B cells", "CD10-\nMature B cells", "CD10-IgD+CD27-\nNaive B cells", "CD10-IgD+CD27-\nDouble Negative B cells", "CD21-\nDN B cells"]

        ax.set_xticklabels(labels, fontsize = 50)

        adjust_box_widths(fig, 0.9)
        h,l = ax.get_legend_handles_labels()
        plt.legend(h[:2],['HEU', 'HEI'], prop={'size': 40}, markerscale=2., title_fontsize = "xx-large", fancybox = True)
        plt.tight_layout()
        plt.show()

  def figure_3c(self):
    #Plotting results of DE in FACS features between HEI and HEU at entry based on results obtained with test_groups()
    base_df = self.df
    base_df_g = base_df.set_index('Age', append=True).set_index("Group", append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")
    time_points = ["entry", 2., 5., 9., 10., 18.]
    
    #DE analysis
    de_final_res = {tp: {} for tp in time_points}
    for tp in de_final_res:
      de_final_res[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], self.hei_df[self.hei_df["Age"] == tp], self.facs_feat(), all = True)
    
    #selection of feature to plot based on DE results
    col= ["Bmem/CD21+IgD+ | Freq. of Mature B cells (%)"]
    
    for t, x in base_df_g.groupby("Age"):
      #selection of entry time point
      if t == "entry":
        #pvalues and pairs for statannotation
        pvalues = []
        pairs = []

        for b in col:
          res_b = ()
          for i in ["HEU", "HEI"]:
            res_b += (b, i)
          pairs.append((res_b))
        pairs = [((i[0], i[1]), (i[0], i[-1])) for i in pairs]

        for i in col:
          pvalues.append(float(de_final_res[t][i]["p-adj"])) #cambiato pvalue con p-adj

        res_box = x.loc[x["feature"].isin(col)]
        fig = plt.figure(figsize = (20, 15)) 

        sns.set_style("white")
        sns.set_context("poster")

        res_box = res_box.astype({"value": float})

        ax = sns.violinplot(x = res_box["feature"],
                            y = res_box["value"],
                            hue = res_box["Group"],
                            hue_order = ["HEU", "HEI"],
                            order = col,
                            inner = None,
                            cut = 0,
                            linewidth = 0)

        plt.setp(ax.collections, alpha=.6)

        ax = sns.swarmplot(x = res_box["feature"],
                           y = res_box["value"],
                           hue = res_box["Group"],
                           hue_order = ["HEU", "HEI"],
                           order = col,
                           dodge = True,
                           s = 10,
                           color = 'k')

        ax = sns.boxplot(showmeans=True,
                  meanline=True,
                  meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                  medianprops={'visible': False},
                  whiskerprops={'visible': False},
                  zorder=10,
                  x="feature",
                  y="value",
                  hue = "Group",
                  hue_order = ["HEU", "HEI"],
                  order = col,
                  data=res_box,
                  showfliers=False,
                  showbox=False,
                  showcaps=False,
                  ax=ax)

        sns.despine(offset=10, trim=True)
        sns.color_palette("Paired")

        plt.xlabel("")
        plt.ylabel("% of CD19+CD20+ B cells", fontsize = 50)
        ax.tick_params(axis='both', which='major', labelsize=55)

        annotator = Annotator(ax, pairs, data = res_box, x = "feature", y = "value", hue = "Group", hue_order = ["HEU", "HEI"], order = col)
        #Formatting p values for plotting purpose
        formatted_pvalues = ["p = 0.007"]

        annotator.set_custom_annotations(formatted_pvalues)
        annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

        ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
        ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
        annotator.annotate()

        labels = ["Unswitched CD21-IgD+CD27+\nMemory B cells"]

        ax.set_xticklabels(labels, fontsize = 50)

        adjust_box_widths(fig, 0.9)
        h,l = ax.get_legend_handles_labels()
        plt.legend(h[:2],['HEU', 'HEI'], prop={'size': 40}, markerscale=2., title_fontsize = "xx-large", fancybox = True)
        plt.tight_layout()
        plt.show()

  def supp_figure_3a(self): 
    fig, ax = plt.subplots(sharex = True, figsize = (37, 15))
    time_points = [9., 10., 18., 19.]
    col = ["Measles Serology"]
    #DE analysis
    de_final_res = {tp: {} for tp in time_points}
    for tp in de_final_res:
      de_final_res[tp] = self.test_groups(self.heu_df[self.heu_df["Age"] == tp], self.hei_df[self.hei_df["Age"] == tp], col, all = True)

    #Archivio differenze ad ogni mese
    g3_de_final_res = {tp: {} for tp in [18., 19.]}

    meas_heu_g3_df = self.heu_g3_df[~self.heu_g3_df["Measles Serology"].isna()]
    meas_hei_g3_df = self.hei_g3_df[~self.hei_g3_df["Measles Serology"].isna()]
    
    #Testo differenze ad ogni mese tra le categorie
    for tp in g3_de_final_res:
      g3_de_final_res[tp] = self.test_groups(meas_heu_g3_df[meas_heu_g3_df["Age"] == tp], meas_hei_g3_df[meas_hei_g3_df["Age"] == tp], ["Measles Serology"], all = True)

    #Testing differences in nog3 group
    nog3_de_final_res = {tp: {} for tp in time_points}

    meas_heu_nog3_df = self.heu_nog3_df[~self.heu_nog3_df["Measles Serology"].isna()]
    meas_heu_nog3_df.replace({'Age' : {"entry": 1., 2.: 1.}}, inplace = True)
    meas_hei_nog3_df = self.hei_nog3_df[~self.hei_nog3_df["Measles Serology"].isna()]
    meas_hei_nog3_df.replace({'Age' : {"entry": 1., 2.: 1.}}, inplace = True)

    for tp in nog3_de_final_res:
      nog3_de_final_res[tp] = self.test_groups(meas_heu_nog3_df[meas_heu_nog3_df["Age"] == tp], meas_hei_nog3_df[meas_hei_nog3_df["Age"] == tp], col)
    
    #Plotting code
    base_df = self.df[self.df["Age"].isin([9., 10., 18., 19.])]

    base_df = base_df[base_df["Measles Serology"].notna()].sort_values(by = ["STUDY ID"])

    base_df.replace({'Age' : {2: "entry"}}, inplace = True)
    base_df.replace({'Age' : {"entry": 1}}, inplace = True)


    base_df["utils"] = base_df[["Group", "EXTRA VACCINATION"]].apply(tuple, axis = 1)

    #pvalues and pairs for statannotation
    pairs = []
    for b in np.sort(base_df["Age"].unique()):
      if "Measles Serology" in de_final_res[b]:
        res_t = ()
        for i in ["HEU", "HEI"]:
          res_t += (b, i)
        pairs.append((res_t))
    pairs = [((i[0], i[1]), (i[0], i[-1])) for i in pairs]

    pvalues = []

    for age in np.sort(base_df["Age"].unique()):
      if "Measles Serology" in de_final_res[age].keys():
        pvalues.append(float(de_final_res[age]["Measles Serology"]["p-value"]))
    
    sns.set_style("white")
    sns.set_context("poster")

    sns.stripplot(data = base_df, x="Age", y="Measles Serology", linewidth = 2, hue= "utils",  hue_order = [('HEU', 0), ('HEU', 1), ('HEI', 0), ('HEI', 1)], palette = ("#1C86EE", "#00CED1", "#FF8C00", "#FFB90F"), dodge = True, size=15, ax = ax)


    pairs_i = []

    for b in np.sort(base_df["Age"].unique()):
      if b >= 9.:
        if "Measles Serology" in nog3_de_final_res[b]:
          res_m = []
          for i in [('HEU', 0), ('HEI', 0)]: #('HEU', 0), ('HEI', 0), ('HEU', 1), ('HEI', 1)    , "('HEU', 1)", "('HEI', 1)"
            res_m.append((b, i))
          pairs_i.append(res_m)
        if b in g3_de_final_res.keys():
          if "Measles Serology" in g3_de_final_res[b]:
            res_m = []
            for i in [('HEU', 1), ('HEI', 1)]:
              res_m.append((b, i))
            pairs_i.append(res_m)

    pairs_i = [(i[0], i[1]) for i in pairs_i]
    pvalues_i = []

    for age in np.sort(base_df["Age"].unique()):
      if age >= 9.:
        if "Measles Serology" in nog3_de_final_res[age].keys():
          pvalues_i.append(float(nog3_de_final_res[age]["Measles Serology"]["p-value"]))
        if age in g3_de_final_res.keys():
          if "Measles Serology" in g3_de_final_res[age].keys():
            pvalues_i.append(float(g3_de_final_res[age]["Measles Serology"]["p-value"]))

    ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Age",
                y="Measles Serology",
                hue = "utils",
                hue_order = [('HEU', 0), ('HEU', 1), ('HEI', 0), ('HEI', 1)],
                data=base_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

    sns.despine(offset=10, trim=True)
    plt.axhline(y=200, color='r', linestyle='--')
    plt.axhline(y=275, color='r', linestyle='--')

    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, 190, "200", color="red", transform=trans,
            ha="right", va="center")
    ax.text(0, 285, "275", color="red", transform=trans,
            ha="right", va="center")

    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 40)
    ax.set_xticklabels(["9", "10", "18", "19"])
    ax.set_ylabel(ax.get_ylabel(), fontsize = 40)
    ax.set_ylim(top = 3700)

    pairs = [((18.0, 'HEU'), (18.0, 'HEI')),
             ((19.0, 'HEU'), (19.0, 'HEI'))]

    pvalues = [0.000565,
               0.000708]

    annotator = Annotator(ax, pairs, data = base_df, x = "Age", y = "Measles Serology", hue = "Group", hue_order = ["HEU", "HEI"], order = [9., 10., 18., 19.], line_offset_to_box = 0.2)
    annotator.set_pvalues(pvalues)
    annotator.configure(loc = "outside", text_format = "full", line_width = 3, fontsize = "medium", show_test_name = False)

    annotator.annotate()

    pairs_i = [((18, ('HEU', 0)), (18, ('HEI', 0))),
    ((18, ('HEU', 1)), (18, ('HEI', 1))),
    ((19, ('HEU', 0)), (19, ('HEI', 0))),
    ((19, ('HEU', 1)), (19, ('HEI', 1)))]
    pvalues_i = [0.04,
                 0.076,
                 0.007,
                 0.0117]

    annotator_2 = Annotator(ax, pairs_i, data = base_df, x = "Age", y = "Measles Serology", hue= "utils", hue_order = [('HEU', 0), ('HEU', 1), ('HEI', 0), ('HEI', 1)], order = [9., 10., 18., 19.], line_offset_to_box = 0.4)
    annotator_2.set_pvalues(pvalues_i)
    annotator_2.configure(loc = "inside", text_format = "full", fontsize = "medium", show_test_name = False)
    annotator_2.annotate()


    h,l = ax.get_legend_handles_labels()
    plt.legend(h[:4],['HEU', 'HEU (e.d.)', "HEI", "HEI (e.d.)"], prop={'size': 25}, markerscale=2.)
    plt.ylabel("Measles Ab mlU/mL", labelpad = 40)
    plt.xlabel("Age in months", labelpad = 40)
    plt.title("Measles Serology in TARA cohort", fontsize = 50, y = 1, pad = 150, fontname="Times New Roman Bold")
    plt.tight_layout()

  def supp_figure_3b(self): #WB score at 9 mnth
    #Testing differences between HEI TTP and HEI TT UP in WB score
    hei_final_res_resp = {}
    feature = ["WB score"]
    time_points = ["entry", 2., 5., 9., 10., 18.]
    feats_resp = self.feats_resp
    hei_final_res_resp = {feat: {} for feat in feats_resp}
    for i in time_points:
      try:
        hei_final_res_resp[feats_resp[0]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[0]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[0]].isin(["UP"])], 
                                                                feature,  
                                                                category = feats_resp[0], 
                                                                all = False)
        hei_final_res_resp[feats_resp[1]][i] = self.test_groups(self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[1]] == "P"], 
                                                                self.hei_df[self.hei_df["Age"] == i].loc[self.hei_df[self.hei_df["Age"] == i][feats_resp[1]].isin(["UP"])], 
                                                                feature,  
                                                                category = feats_resp[1], 
                                                                all = False)
      except ValueError:
        continue
    print(hei_final_res_resp)
    #steps to find paired data between entry and 9 months 
    feature = "TT Response"

    paired = self.hei_df[self.hei_df["Age"].isin(["entry", 9.])]

    list_common_id = []
    for val, val_df in paired.groupby("Age"):
      list_common_id.append(list(val_df["STUDY ID"].unique()))

    paired = paired[paired["STUDY ID"].isin(list(set.intersection(*map(set,list_common_id))))]
    #Testing differences between HEI TTP and HEI TT UP in WB score on paired samples
    wb_final_res = {gp: {} for gp in ["P", "UP"]}  
    for gp in ["P", "UP"]:
      wb_final_res[gp] = self.test_groups(paired[paired["Age"] == "entry"][paired[paired["Age"] == "entry"][feature] == gp], paired[paired["Age"] == 9.][paired[paired["Age"] == 9.][feature] == gp], ["WB score"], "SW", all = False)

    #Plotting part
    base_df = self.hei_df[self.hei_df["Age"].isin(["entry", 9.])]
    base_df_g = base_df.set_index('Age', append=True).set_index(feature, append = True).stack().to_frame().reset_index().rename(columns = {"level_3": "feature", 0: "value"}).drop("level_0", axis = "columns")

    col = ["WB score"]
    #saving pairings and pvalues for statannotation
    pvalues = []
    pairs = []

    for b in base_df["Age"].unique():
      if col[0] in hei_final_res_resp[feature][b]: 
        res_b = ()
        for i in ["UP", "P"]:
          res_b += (b, i)
        pairs.append((res_b))

    for pairings in [(base_df["Age"].unique()[0], "UP", base_df["Age"].unique()[1], "UP"), (base_df["Age"].unique()[0], "P", base_df["Age"].unique()[1], "P")]:
      pairs.append(pairings)

    pairs = list(map(lambda x: ((x[0], x[1]), (x[2], x[3])), pairs))
    for t in base_df["Age"].unique():
      for i in col:
        if i in hei_final_res_resp[feature][t]: 
          pvalues.append(float(hei_final_res_resp[feature][t][i]["p-adj"]))

    for cat in ["UP", "P"]:
      pvalues.append(float(wb_final_res[cat]["WB score"]["p-value"]))

    res_box = base_df_g.loc[base_df_g["feature"].isin(col)]

    fig = plt.figure(figsize = (21,21))

    sns.set_style("white")
    sns.set_context("poster")

    res_box = res_box.astype({"value": float})
    ax = sns.violinplot(x = res_box["Age"], y = res_box["value"], hue = res_box[feature], hue_order = ["UP", "P"], order = ["entry", 9.], inner = None, cut = 0, linewidth = 0, palette = ["#FFCBA4", "#FF6700"])

    plt.setp(ax.collections, alpha=.6)

    ax = sns.swarmplot(x = res_box["Age"], y = res_box["value"], hue = res_box[feature], hue_order = ["UP", "P"], order = ["entry", 9.], dodge = True, s = 10, color = 'k')

    ax = sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '--', 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="Age",
                y="value",
                hue = feature,
                hue_order = ["UP", "P"],
                order = ["entry", 10.],
                data=res_box,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=ax)

    sns.despine(offset=10, trim=True)
    sns.color_palette("Paired")

    plt.xlabel("Age in months", fontsize = 55)
    plt.ylabel("WB score", fontsize = 55)
    ax.tick_params(axis='both', which='major', labelsize=55)
    ax.tick_params(axis='both', which='minor', labelsize=35)
    ax.set_xticklabels(['1-2', '9'])


    annotator = Annotator(ax, pairs, data = res_box, x = "Age", y = "value", hue = feature, hue_order = ["UP", "P"], order = ["entry", 9.])
    formatted_pvalues = [ "p = 0.03", "p = 0.001", "p = 0.005"]

    annotator.set_custom_annotations(formatted_pvalues)
    #annotator.set_pvalues(pvalues)

    annotator.configure(text_format = "full", fontsize = "xx-large", show_test_name = False)

    ax.set_xlabel(ax.get_xlabel(),labelpad = 40)
    ax.set_ylabel(ax.get_ylabel(), labelpad = 40)
    annotator.annotate()

    adjust_box_widths(fig, 0.9)
    h,l = ax.get_legend_handles_labels()
    plt.legend(h[:2],['HEI TT UP', 'HEI TT P'], prop={'size': 30}, markerscale=2., title_fontsize = "xx-large", fancybox = True, loc = "best")
    fig.show()

  def supp_figure_3c(self):
    #Durability of TT Response in HEI and HEU
    #selection of data based on time points 
    data_heu = self.heu_df.dropna(subset=['Tetanous Serology'])
    data_heu.rename(columns={"Tetanous Serology": "Tetanous_Serology"}, inplace=True)
    data_heu = data_heu[data_heu["Age"].isin([5., 9., 10., 18.])]
    data_heu["Age"] = pd.to_numeric(data_heu["Age"])
    data_hei = self.hei_df.dropna(subset=['Tetanous Serology'])
    data_hei.rename(columns={"Tetanous Serology": "Tetanous_Serology"}, inplace=True)
    data_hei = data_hei[data_hei["Age"].isin([5., 9., 10., 18.])]
    data_hei["Age"] = pd.to_numeric(data_hei["Age"])

    # Combine the dataframes for analysis 
    data = pd.concat([data_heu, data_hei], axis=0, ignore_index=True)
    data = data.dropna(subset=['Tetanous_Serology'])
    data = data[data["Age"] != "entry"]
    data["Age"] = pd.to_numeric(data["Age"])

    #compute mixed effect model
    mixed_effects_model_heu = sm.MixedLM.from_formula("Tetanous_Serology ~ Age", groups='STUDY ID', data=data_heu)
    result_heu = mixed_effects_model_heu.fit()
    mixed_effects_model_hei = sm.MixedLM.from_formula("Tetanous_Serology ~ Age", groups='STUDY ID', data=data_hei)
    result_hei = mixed_effects_model_hei.fit()

    #compute pearson correlation
    corr_heu, pval_heu = stats.pearsonr(data_heu["Age"], data_heu["Tetanous_Serology"])
    slope_heu = result_heu.params["Age"]
    intercept_heu = result_heu.params["Intercept"]

    corr_hei, pval_hei = stats.pearsonr(data_hei["Age"], data_hei["Tetanous_Serology"])
    slope_hei = result_hei.params["Age"]
    intercept_hei = result_hei.params["Intercept"]

    x_heu = np.linspace(data_heu['Age'].min(), data_heu['Age'].max(), 100)
    x_hei = np.linspace(data_hei['Age'].min(), data_hei['Age'].max(), 100)
    y_heu = result_heu.predict(exog=dict(Age=x_heu))
    y_hei = result_hei.predict(exog=dict(Age=x_hei))

    #plotting code
    fig, ax = plt.subplots(figsize = (10,10))
    data_heu.plot(kind='scatter', x='Age', y='Tetanous_Serology', ax=ax, color='blue')
    data_hei.plot(kind='scatter', x='Age', y='Tetanous_Serology', ax=ax, color='orange')

    text_heu = f'HEU r = {corr_heu:.2f}\nHEU p = {pval_heu:.2e}\nHEU slope = {slope_heu:.2f}\nHEU intercept = {intercept_heu:.2f}'
    text_hei = f'HEI r = {corr_hei:.2f}\nHEI p = {pval_hei:.2e}\nHEI slope = {slope_hei:.2f}\nHEI intercept = {intercept_hei:.2f}'

    ax.text(0.25, 0.95, text_heu, transform=ax.transAxes, va='top', size = 12)
    ax.text(0.5, 0.95, text_hei, transform=ax.transAxes, va='top', size = 12)

    ax.plot(x_heu, y_heu, color='blue')
    ax.plot(x_hei, y_hei, color='orange')

    ax.set_xlabel('Age (months)', fontsize = 20, labelpad = 20)
    ax.set_xticks([5, 9, 10, 18])
    ax.set_ylabel('Tetanous Serology (IU/mL)', fontsize = 25, labelpad = 20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_title('Durability of Tetanous Serology', pad = 50, fontsize = 30)
    ax.legend(['HEU', "HEI", 'LMM fit HEU', 'LMM fit HEI'], prop={'size': 15}, markerscale=2., title_fontsize = "xx-large", fancybox = True, loc = "upper right")
    plt.tight_layout()
    plt.show()

  def supp_figure_entry_corr(self):
    #Correlations between entry and 5-10 or 18 months TT Serology (paired patients)
    # filter data to include only "entry", 10 and 18 years old
    ages = ["entry", 5, 10, 18]
    #selection of data based on time points and column renaming
    data_heu = self.heu_df.dropna(subset=['Tetanous Serology'])
    data_heu.rename(columns={"Tetanous Serology": "Tetanous_Serology"}, inplace=True)
    data_hei = self.hei_df.dropna(subset=['Tetanous Serology'])
    data_hei.rename(columns={"Tetanous Serology": "Tetanous_Serology"}, inplace=True)

    data_age_heu = data_heu[data_heu["Age"].isin(ages)]
    data_age_hei = data_hei[data_hei["Age"].isin(ages)]

    # group data by "STUDY ID"
    data_grouped_heu = data_age_heu.groupby("STUDY ID")
    data_grouped_hei = data_age_hei.groupby("STUDY ID")

    # create an empty list to store the correlation coefficients
    correlations_heu = []
    correlations_hei = []

    # loop through each group and compute the correlation between "entry" and "Age" = 10 and 18
    paired_values_5_heu = []
    paired_values_10_heu = []
    paired_values_18_heu = []
    paired_values_5_hei = []
    paired_values_10_hei = []
    paired_values_18_hei = []

    for group in data_grouped_heu:
      # get the values for "entry" and "Age" = 10 and 18
      values_entry = group[1][group[1]["Age"] == "entry"]["Tetanous_Serology"].values
      values_5 = group[1][group[1]["Age"] == 5]["Tetanous_Serology"].values
      values_10 = group[1][group[1]["Age"] == 10]["Tetanous_Serology"].values
      values_18 = group[1][group[1]["Age"] == 18]["Tetanous_Serology"].values
    
      # get the paired values for "entry" and "Age" = 10
      for idx in group[1].index:
        if group[1].loc[idx, "Age"] == "entry":
            study_id = group[1].loc[idx, "STUDY ID"]
            entry_value = group[1].loc[idx, "Tetanous_Serology"]
            try:
                #save paired values
                age_5_value = group[1][(group[1]["STUDY ID"] == study_id) & (group[1]["Age"] == 5)]["Tetanous_Serology"].values[0]
                paired_values_5_heu.append((entry_value, age_5_value))
            except IndexError:
                pass
            try:
                #save paired values
                age_10_value = group[1][(group[1]["STUDY ID"] == study_id) & (group[1]["Age"] == 10)]["Tetanous_Serology"].values[0]
                paired_values_10_heu.append((entry_value, age_10_value))
            except IndexError:
                pass
            try:
                #save paired values
                age_18_value = group[1][(group[1]["STUDY ID"] == study_id) & (group[1]["Age"] == 18)]["Tetanous_Serology"].values[0]
                paired_values_18_heu.append((entry_value, age_18_value))
            except IndexError:
                pass

    for group_hei in data_grouped_hei:
    
      # get the values for "entry" and "Age" = 10 and 18
      values_entry = group_hei[1][group_hei[1]["Age"] == "entry"]["Tetanous_Serology"].values
      values_5 = group_hei[1][group_hei[1]["Age"] == 5]["Tetanous_Serology"].values
      values_10 = group_hei[1][group_hei[1]["Age"] == 10]["Tetanous_Serology"].values
      values_18 = group_hei[1][group_hei[1]["Age"] == 18]["Tetanous_Serology"].values
    
      # get the paired values for "entry" and "Age" = 10
      for idx in group_hei[1].index:
        if group_hei[1].loc[idx, "Age"] == "entry":
            study_id = group_hei[1].loc[idx, "STUDY ID"]
            entry_value = group_hei[1].loc[idx, "Tetanous_Serology"]
            try:
                age_5_value = group_hei[1][(group_hei[1]["STUDY ID"] == study_id) & (group_hei[1]["Age"] == 5)]["Tetanous_Serology"].values[0]
                paired_values_5_hei.append((entry_value, age_5_value))
            except IndexError:
                pass
            try:
                age_10_value = group_hei[1][(group_hei[1]["STUDY ID"] == study_id) & (group_hei[1]["Age"] == 10)]["Tetanous_Serology"].values[0]
                paired_values_10_hei.append((entry_value, age_10_value))
            except IndexError:
                pass
            try:
                age_18_value = group_hei[1][(group_hei[1]["STUDY ID"] == study_id) & (group_hei[1]["Age"] == 18)]["Tetanous_Serology"].values[0]
                paired_values_18_hei.append((entry_value, age_18_value))
            except IndexError:
                pass

    # compute the correlation coefficients between "entry" and "Age" = 10 and 18
    corr_5_heu, pval_5_heu = pearsonr([x[0] for x in paired_values_5_heu], [x[1] for x in paired_values_5_heu])
    corr_10_heu, pval_10_heu = pearsonr([x[0] for x in paired_values_10_heu], [x[1] for x in paired_values_10_heu])
    corr_18_heu, pval_18_heu = pearsonr([x[0] for x in paired_values_18_heu], [x[1] for x in paired_values_18_heu])
    corr_5_hei, pval_5_hei = pearsonr([x[0] for x in paired_values_5_hei], [x[1] for x in paired_values_5_hei])
    corr_10_hei, pval_10_hei = pearsonr([x[0] for x in paired_values_10_hei], [x[1] for x in paired_values_10_hei])
    corr_18_hei, pval_18_hei = pearsonr([x[0] for x in paired_values_18_hei], [x[1] for x in paired_values_18_hei])

    #plotting code
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30,20))

    for i, ax in enumerate(axs.flatten()[:3]):
      age = [5, 10, 18][i]
      paired_values = eval(f'paired_values_{age}_heu')
      corr, pval = eval(f'corr_{age}_heu'), eval(f'pval_{age}_heu')

      # Plot the scatter plots
      ax.scatter([x[0] for x in paired_values], [x[1] for x in paired_values], s = 60)

      # Add the linear regression lines
      m, b = np.polyfit([x[0] for x in paired_values], [x[1] for x in paired_values], 1)
      ax.plot([min([x[0] for x in paired_values]), max([x[0] for x in paired_values])], 
            [m*min([x[0] for x in paired_values])+b, m*max([x[0] for x in paired_values])+b], 
            color = "red", label=f"Age {age}: y={m:.2f}x+{b:.2f}")

      # Add the correlation coefficient
      ax.text(0.8, 0.9, f"r= {corr:.2f}\np={pval:.2f}", transform=ax.transAxes, va='top', size = 20)

      # Add the x and y labels
      ax.set_xlabel("Tetanous Serology at Entry", fontsize = 20, labelpad = 10)
      ax.set_ylabel(f"Tetanous Serology at Age {age}", fontsize = 20, labelpad = 10)
      ax.tick_params(axis='both', which='major', labelsize=20)
      ax.tick_params(axis='both', which='minor', labelsize=20)

      # Add a legend
      ax.legend(prop={'size': 20}, markerscale=1., title_fontsize = "xx-large", fancybox = True, loc = "upper right")
    
    for i, ax in enumerate(axs.flatten()[3:]):
      age = [5, 10, 18][i]
      paired_values = eval(f'paired_values_{age}_hei')
      corr, pval = eval(f'corr_{age}_hei'), eval(f'pval_{age}_hei')

      # Plot the scatter plots
      ax.scatter([x[0] for x in paired_values], [x[1] for x in paired_values], color = "orange", s = 60)

      # Add the linear regression lines
      m, b = np.polyfit([x[0] for x in paired_values], [x[1] for x in paired_values], 1)
      ax.plot([min([x[0] for x in paired_values]), max([x[0] for x in paired_values])], 
              [m*min([x[0] for x in paired_values])+b, m*max([x[0] for x in paired_values])+b], 
              color = "red", label=f"Age {age}: y={m:.2f}x+{b:.2f}")

      # Add the correlation coefficient
      ax.text(0.8, 0.9, f"r= {corr:.2f}\np={pval:.2f}", transform=ax.transAxes, va='top', size = 20)

      # Add the x and y labels
      ax.set_xlabel("Tetanous Serology at Entry", fontsize = 20, labelpad = 10)
      ax.set_ylabel(f"Tetanous Serology at Age {age}", fontsize = 20, labelpad = 10)
      ax.tick_params(axis='both', which='major', labelsize=20)
      ax.tick_params(axis='both', which='minor', labelsize=20)

      # Add a legend
      ax.legend(prop={'size': 20}, markerscale=1., title_fontsize = "xx-large", fancybox = True, loc = "upper right")

  def supp_figure_PWC(self):
    #Power calculation based on differences in TT Serology at 5 months
    # HEU meand and quartiles
    Amd = self.heu_df[self.heu_df["Age"] == 5.]["Tetanous Serology"].mean()
    Aq1 = np.percentile(self.heu_df[(self.heu_df["Age"] == 5.) & (~self.heu_df["Tetanous Serology"].isna())]["Tetanous Serology"], 25)
    Aq3 = np.percentile(self.heu_df[(self.heu_df["Age"] == 5.) & (~self.heu_df["Tetanous Serology"].isna())]["Tetanous Serology"], 75)

    # HEI mean and quartiles
    Bmd = self.hei_df[self.hei_df["Age"] == 5.]["Tetanous Serology"].mean()
    Bq1 = np.percentile(self.hei_df[(self.hei_df["Age"] == 5.) & (~self.hei_df["Tetanous Serology"].isna())]["Tetanous Serology"], 25)
    Bq3 = np.percentile(self.hei_df[(self.hei_df["Age"] == 5.) & (~self.hei_df["Tetanous Serology"].isna())]["Tetanous Serology"], 75)

    # Mean
    Am = (Aq1 + Amd + Aq3)/3
    Bm = (Bq1 + Bmd + Bq3)/3

    # SD
    Asd = (Aq3 - Aq1)/1.35
    Bsd = (Bq3 - Bq1)/1.35

    # Cohen's d = (M2 - M1) ⁄ SDpooled 
    SDpooled = ((Asd**2 + Bsd**2)/2)**0.5 # SDpooled = √((SD1^2 + SD2^2) ⁄ 2)

    # Calculate Cohen's d and perform a two-tailed t-test with alpha=0.01 and power=pwlim/100
    effsize = pg.compute_effsize(self.heu_df[(self.heu_df["Age"] == 5.) & (~self.heu_df["Tetanous Serology"].isna())]["Tetanous Serology"], 
                                 self.hei_df[(self.hei_df["Age"] == 5.) & (~self.hei_df["Tetanous Serology"].isna())]["Tetanous Serology"], 
                                 paired=False, 
                                 eftype='cohen')

    sns.set(style='ticks', context='notebook', font_scale=1.2)
    d = effsize  # Fixed effect size
    n = np.arange(5, 80, 5)  # Incrementing sample size
    # Compute the achieved power
    pwr = pg.power_ttest(d=d, n=n, contrast='two-samples')
    # Start the plot
    plt.plot(n, pwr, 'ko-.')
    plt.axhline(0.8, color='r', ls=':')
    plt.xlabel('Sample size')
    plt.ylabel('Power (1 - type II error)')
    plt.title('Achieved power of a paired T-test')
    sns.despine()
    plt.show()
    
   
                                                           ######################################
                                                           #               USAGE                #
                                                           ######################################

         
# PREPARING INPUT DATA
metaFeatures = list(range(3, 41)) #columns with metadata
catFeatures = [2] #(sex)

r01_base = "R01_DB.xlsx"
path = ""

#initializing class
analysis = Analyzer(r01_base, metaFeatures, catFeatures)

#Plots in manuscript
analysis.figure_1a()
analysis.figure_1b()
analysis.figure_1c()
analysis.figure_1d()
analysis.figure_1e()
analysis.figure_1f()
analysis.figure_1g()
analysis.figure_2a()
analysis.figure_2b()
analysis.figure_3a()
analysis.figure_3b()
analysis.figure_3c()

#create dataframe for R PCA analysis
analysis.make_pca_df(save = True, outFile = f"{path}R01_DB_PCA.xlsx")

#model training, computationally expensive
analysis.train_nested_xgb(outFile = f"{path}model_xgb")    

supplementay_table_2 = analysis.supp_table_2()
analysis.supp_figure_3a()
analysis.supp_figure_3b()
analysis.supp_figure_3c()
analysis.supp_figure_PWC()
analysis.supp_figure_entry_corr()

#OPENING FINAL TRAINED MODEL AND PLOTTING RESULTS
with open(f"{path}model_xgb_final.sav", 'rb') as f:
  model = pickle.load(f)

df = analysis.make_ml_dataset()
X = df.drop(columns = ["TT Response"], inplace = False)
y = df["TT Response"]


results = model_selection.cross_val_score(model, X[['Bmem/CD21-IgD-/CD25 %', 'unswMem/IgG-IgM+ %']], y, cv = LeaveOneOut())
print("accuracy :")
print(results.mean())
print(results.std())

#Figure 4 panel 2 top
analysis.test_ml_algorithms(save = True, outFile = "f{path}lazypredict_ml_tests.png")
#Figure 4 panel 2 bottom
analysis.plot_model_feat(f"{path}model_xgb_final.sav", X=X[['Bmem/CD21-IgD-/CD25 %', 'unswMem/IgG-IgM+ %']], y = y, save=True, outName=f"{path}")
