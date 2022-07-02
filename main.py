# # Class Imbalance Study
# This study is a reaction to the work by Ruben van den Goorbergh, Maarten van Smeden, Dirk Timmerman, and Ben Van Calster, available here: [The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression](https://doi.org/10.1093/jamia/ocac093)

# # Objective
# I want to use synthetic data to examine:
#  - Are class-balancing methods harmful to model performance?
#  - When should class-balancing methods be used?

# # Methods

#

# # Experiments

# ## Load Libraries

# +
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import run_experiment as run_exp
#import treatments as trmts
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
#import metrics as mtrx
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
#import models as mdls
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# -

# ## Configure experiments

# +
# Measurement
metrics = [roc_auc_score, matthews_corrcoef, f1_score]

# Parameters
models = [LogisticRegression(), DecisionTreeClassifier()]
treatments = [FunctionSampler(), RandomOverSampler(), RandomUnderSampler()]#, SMOTE()]
imbalances = [0.5, 0.25, 0.1, 0.05, 0.01, 0.001]
classes = [2]
ns = [10000]

# Cross-product of the parameters
experiments_list = []

for t in treatments:
    for m in models:
        for c in classes:
            for n in ns:
                for i in imbalances:
                    experiments_list.append({
                        "model": m,
                        "metrics": metrics,
                        "treatment": t,
                        "imbalance": i,
                        "classes": c,
                        "n": n,
                        "name": f"{m.__class__.__name__} with {t.__class__.__name__} on {c} classes, having {i} minority classes, N={n}"
                        })
# -

print(f"{len(experiments_list)} experiments prepared.")

# ## Run experiments

# +
trials = 10

for i, config in enumerate(experiments_list):
    print(f"Running experiment {i} of {len(experiments_list)} ({config['name']})")
    result = None
    for trial in range(0, trials):
        trial_result, cf_matrix = run_exp.run(config)
        # Record the result
        if result is None:
            result = trial_result
        else:
            for m_i in range(0, len(config["metrics"])):
                result[m_i] += trial_result[m_i]
    # Average the results across trials
    for m_i in range(0, len(config["metrics"])):
        result[m_i] = result[m_i]/trials
    experiments_list[i]["result"] = result
    experiments_list[i]["cf_matrix"] = cf_matrix
    #print(result)
# -

# # Results

# ### Performance metrics by model and treatment

#plt.figure(1)
#plt.subplot(len(metrics), len(models), 1)
plot_num = 1
for mo_i, mo in enumerate(models):
    for m_i, m in enumerate(metrics):
        #plt.subplot(len(metrics), len(models), plot_num)
        plot_num += 1
        plot_data = [experiment["result"][m_i] for experiment in experiments_list if experiment["model"]==mo]
        plot_data = pd.DataFrame({m.__name__: plot_data})
        plot_data["Imbalance"] = [experiment["imbalance"] for experiment in experiments_list if experiment["model"]==mo]
        plot_data["Treatment"] = [experiment["treatment"].__class__.__name__ for experiment in experiments_list if experiment["model"]==mo]
        plot_data = plot_data.pivot("Imbalance", "Treatment", m.__name__)
        plt.figure(plot_num)
        sns_plt = sns.lineplot(data=plot_data)
        sns_plt.set(ylabel=m.__name__)
        sns_plt.invert_xaxis()
        sns_plt.set_title(f"{m.__name__} vs imbalance for {models[mo_i].__class__.__name__}")

# ### Confusion matrices in High-Imbalance Case (minority class = 0.1%)

high_imb_exp = [experiment for experiment in experiments_list if experiment["imbalance"]==0.001]
for mo_i, mo in enumerate(models):
    for exp in high_imb_exp:
        plot_num += 1
        plt.figure(plot_num)
        sns_plt = sns.heatmap(exp["cf_matrix"], annot=True, cmap='OrRd')
        sns_plt.set_title(f"{models[mo_i].__class__.__name__} {exp['treatment'].__class__.__name__} confusion matrix")

# # Analysis

# ## Are class-balancing methods harmful to model performance?

# ### Class-balancing vs threshold tuning for F1



# ### Class-balancing vs threshold tuning for MCC



# ## When should class-balancing methods be used?

# ### Moderate imbalance (10%)



# ### High imbalance (0.1%)



# # Conclusion



# # References

# ### 1
# Ruben van den Goorbergh, Maarten van Smeden, Dirk Timmerman, Ben Van Calster, The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression, Journal of the American Medical Informatics Association, 2022;, ocac093, https://doi.org/10.1093/jamia/ocac093
