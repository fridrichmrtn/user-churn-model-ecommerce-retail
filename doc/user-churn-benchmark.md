> Martin Fridrich, 03/2021

# Predicting user churn with various classifiers

This document strives to propose, implement, and benchmark machine learning solutions to the user churn prediction problem. For such a task, we utilize original user churn model formed in the previous step. The endevours are structured as follows:

1 [Housekeepin'](#housekeepin)  
2 [Classification pipeline](#classification-pipeline)  
3 [Experimental design & fitting](#experimental-design--fitting)  
4 [Overview](#overview)  
5 [Bias-variance tradeoff](#bias-variance-tradeoff)  
4 [ROC-AUC curves](#roc-auc-curves)  
5 [Next steps](#next-steps)  

# Housekeepin'

In the opening section, we load most of the libs and the `user-churn-model-data.csv`. Also, we downcasted the numerical columns.


```python
# set options
import warnings  
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option("notebook_repr_html", False)

# general
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from itertools import permutations
from collections import OrderedDict
import pickle

# plotting
from cairosvg import svg2png
from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns

# pipes & pints
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

# preprocessing & dr
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# clfs
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# others
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import  Nystroem
plt.rcParams['figure.dpi'] = 450
plt.rcParams['savefig.dpi'] = 450

# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score,\
f1_score, roc_auc_score, roc_curve
```


```python
def downcast_dtype(column_series):
    """Try to infer better dtype.
    
        Parameters>
            column_series: pd.Series, a column to down
            
        Returns>
            pd.Series, converted column"""        
    
    conv_dict = {"int":"integer", "float":"float"}
    for k in conv_dict.keys():
        if k in str(column_series.dtype):
            return pd.to_numeric(column_series, downcast=conv_dict[k])
```


```python
# data load
classification_data = pd.read_csv("../data/ecom-user-churn-data.csv")
classification_data = classification_data.loc[:,"ses_rec":"target_class"]
classification_data = classification_data.apply(downcast_dtype, axis=0)
features = classification_data.loc[:,"ses_rec":"int_cat24_n"].columns
```

# Classification pipeline

We tackle the user churn prediction problem with a machine learning pipeline, which consists of the following blocks (1) data processing, (2) feature extraction, and (3) classification. We remove explanatory variables with low variance in the first block, scaled them with quantile transformer, and extend the original feature space with 2nd degree polynoms & interactions. For the second block, we use principal component analysis to remove autocorrelation. The last block is dedicated to the classification with logistic regression, multi-layer perceptron, support vector machine & ensembles.


```python
tran = [("nzv", VarianceThreshold()), ("scale",QuantileTransformer()),
    ("poly", PolynomialFeatures()), ("dr", PCA(n_components=50))]

clf = {"lr":LogisticRegression(solver="liblinear"), "mlp":MLPClassifier(), "svm-lin":CalibratedClassifierCV(LinearSVC()),
    "svm-rbf":[("rbf", Nystroem(random_state=2021)), ("clf", CalibratedClassifierCV(LinearSVC()))], # speed up the rbf
    "rf":RandomForestClassifier(), "gbm":GradientBoostingClassifier()}

pipelines = OrderedDict((k, Pipeline(tran+c)) if isinstance(c, list) else (k, Pipeline(tran+[("clf",c)])) for k,c in clf.items())
```

# Experimental design & fitting

To evaluate the pipelines, we employ a robust procedure based on stratified 20-fold cross-validation, where pipeline performance is evaluated with respect to accuracy, precision, recall, f1, roc_auc, and fitting time. 


```python
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=2021)
scoring_metrics = {'acc': "accuracy", "pre":"precision","rec":"recall",
    "f1":"f1", "auc":"roc_auc"}

score_dict = {k:cross_validate(p,
    X=classification_data.loc[:,features],
    y=classification_data.target_class, 
    cv=skf,
    scoring=scoring_metrics, n_jobs=8, return_train_score=True,
    return_estimator=True, error_score=0)
        for k,p in pipelines.items()}

for k in score_dict.keys():
    score_dict[k]["pipe"] = np.repeat(k, skf.get_n_splits())
    
score_df = pd.concat([pd.DataFrame.from_dict(s)
    for s in score_dict.values()]) 
```

# Overview

In the following chunk we compute point estimates and confidence intervals for the performance metrics and computational time.


```python
def get_ci(df, ignore):
    cols = [c for c in df.columns if c not in set(ignore)]
    mju, scaler = df[cols].mean().values, df[cols].std()/np.sqrt(df.shape[0])
    hi, low = mju + 1.96 * scaler, mju - 1.96 * scaler
    output_dict = dict(**dict(zip(["mju_"+ c for c in cols], mju)),
        **dict(zip(["hi_"+ c for c in cols], hi)),
        **dict(zip(["low_"+ c for c in cols], low)))
    return pd.Series(output_dict,
                index=output_dict.keys())

pipe_order = pipelines.keys()

score_df.groupby("pipe").apply(get_ci, ignore=["pipe", "estimator"]).to_csv("../data/ecom-user-churn-bench.csv")
score_df.groupby("pipe").apply(get_ci, ignore=["pipe", "estimator"]).loc[:,
    ["mju_test_acc", "mju_test_pre", "mju_test_rec", "mju_test_f1", "mju_test_auc", "mju_fit_time"]].\
    loc[pipe_order,:].reset_index()
```




          pipe  mju_test_acc  mju_test_pre  mju_test_rec  mju_test_f1  \
    0       lr      0.889420      0.891738      0.996065     0.941018   
    1      mlp      0.876838      0.898702      0.970305     0.933122   
    2  svm-lin      0.888853      0.892722      0.993937     0.940614   
    3  svm-rbf      0.887880      0.892423      0.993114     0.940079   
    4       rf      0.886888      0.894261      0.989247     0.939358   
    5      gbm      0.889359      0.893214      0.993892     0.940866   
    
       mju_test_auc  mju_fit_time  
    0      0.737367      5.882586  
    1      0.682340     43.507699  
    2      0.731530    112.710903  
    3      0.734517     14.438497  
    4      0.710412     69.696305  
    5      0.742694    107.929758  



From the point estimates, we can see that prediction performance on test part of the data is acceptable and comparable across the classifiers. The promising results are yielded by `lr`, `svm-rbf` & `gbm`, while `lr` being 2-20 times faster. In addition to the plain point estimates, we employ a t-test and couple the observations on cv folds to compare the results. Tests will be evaluated on raw alpha 0.01 and further adjusted with Bonferroni correction.


```python
# do tests on f1 & auc roc
perms = list(permutations(score_df["pipe"].unique(),2))
dims = ["test_acc", "test_f1", "test_auc"]
test_ls = []
for tc in perms:
    for td in dims:
        cf = [cf for cf in score_df.columns if td in cf]
        stat, pval = ttest_rel(score_df.loc[score_df["pipe"]==tc[0],cf].values[:,0],
            score_df.loc[score_df["pipe"]==tc[1],cf].values[:,0])
        mju0 = score_df.loc[score_df["pipe"]==tc[0],cf].values.mean()
        mju1 = score_df.loc[score_df["pipe"]==tc[1],cf].values.mean()
        test_ls.append([tc[0], tc[1], td.split("_")[0], td.split("_")[1], mju0-mju1, pval])
test_df = pd.DataFrame(test_ls, columns=["from", "to", "set","metric", "diff", "pval"])
test_df.to_csv("../data/ecom-user-churn-bench-tests.csv")

# peek at the statistically significant diffs on adjusted p-val
test_df[(test_df.pval*test_df.shape[0]/2<0.01)]
```




           from       to   set metric      diff          pval
    0        lr      mlp  test    acc  0.012582  1.956199e-09
    1        lr      mlp  test     f1  0.007895  3.561566e-10
    2        lr      mlp  test    auc  0.055027  1.084586e-11
    5        lr  svm-lin  test    auc  0.005837  4.298028e-07
    9        lr       rf  test    acc  0.002533  1.139482e-04
    10       lr       rf  test     f1  0.001660  1.275287e-05
    11       lr       rf  test    auc  0.026955  6.030084e-08
    15      mlp       lr  test    acc -0.012582  1.956199e-09
    16      mlp       lr  test     f1 -0.007895  3.561566e-10
    17      mlp       lr  test    auc -0.055027  1.084586e-11
    18      mlp  svm-lin  test    acc -0.012014  8.595428e-10
    19      mlp  svm-lin  test     f1 -0.007491  1.783977e-10
    20      mlp  svm-lin  test    auc -0.049190  2.812392e-11
    21      mlp  svm-rbf  test    acc -0.011042  2.908814e-09
    22      mlp  svm-rbf  test     f1 -0.006956  5.665764e-10
    23      mlp  svm-rbf  test    auc -0.052177  1.838265e-12
    24      mlp       rf  test    acc -0.010049  2.549489e-09
    25      mlp       rf  test     f1 -0.006235  6.182400e-10
    26      mlp       rf  test    auc -0.028072  1.095914e-06
    27      mlp      gbm  test    acc -0.012521  2.237204e-10
    28      mlp      gbm  test     f1 -0.007744  5.082953e-11
    29      mlp      gbm  test    auc -0.060354  6.059465e-13
    32  svm-lin       lr  test    auc -0.005837  4.298028e-07
    33  svm-lin      mlp  test    acc  0.012014  8.595428e-10
    34  svm-lin      mlp  test     f1  0.007491  1.783977e-10
    35  svm-lin      mlp  test    auc  0.049190  2.812392e-11
    40  svm-lin       rf  test     f1  0.001256  1.268514e-04
    41  svm-lin       rf  test    auc  0.021118  6.469678e-07
    44  svm-lin      gbm  test    auc -0.011164  1.789375e-07
    48  svm-rbf      mlp  test    acc  0.011042  2.908814e-09
    49  svm-rbf      mlp  test     f1  0.006956  5.665764e-10
    50  svm-rbf      mlp  test    auc  0.052177  1.838265e-12
    56  svm-rbf       rf  test    auc  0.024105  7.201590e-10
    59  svm-rbf      gbm  test    auc -0.008177  1.974536e-04
    60       rf       lr  test    acc -0.002533  1.139482e-04
    61       rf       lr  test     f1 -0.001660  1.275287e-05
    62       rf       lr  test    auc -0.026955  6.030084e-08
    63       rf      mlp  test    acc  0.010049  2.549489e-09
    64       rf      mlp  test     f1  0.006235  6.182400e-10
    65       rf      mlp  test    auc  0.028072  1.095914e-06
    67       rf  svm-lin  test     f1 -0.001256  1.268514e-04
    68       rf  svm-lin  test    auc -0.021118  6.469678e-07
    71       rf  svm-rbf  test    auc -0.024105  7.201590e-10
    72       rf      gbm  test    acc -0.002472  9.135735e-06
    73       rf      gbm  test     f1 -0.001508  1.455383e-06
    74       rf      gbm  test    auc -0.032282  3.064058e-11
    78      gbm      mlp  test    acc  0.012521  2.237204e-10
    79      gbm      mlp  test     f1  0.007744  5.082953e-11
    80      gbm      mlp  test    auc  0.060354  6.059465e-13
    83      gbm  svm-lin  test    auc  0.011164  1.789375e-07
    86      gbm  svm-rbf  test    auc  0.008177  1.974536e-04
    87      gbm       rf  test    acc  0.002472  9.135735e-06
    88      gbm       rf  test     f1  0.001508  1.455383e-06
    89      gbm       rf  test    auc  0.032282  3.064058e-11



The test results support the observation made with the previous data. There appears to be significant difference between the `svm-rbf`, and `gbm` with wrt auc.

# Bias-variance tradeoff

Now, let us inspect the generalization ability of the classifiers.


```python
fig, axs = plt.subplots(1,3, figsize=(21,7))
mets = {"acc":(0.8,1.005), "f1":(0.9,1.005), "auc":(0.5,1.005)}
pipe_order = ["lr", "svm-lin", "svm-rbf", "mlp", "rf", "gbm"]

for m, ax in zip(mets.keys(), axs):
    sns.scatterplot("train_"+m,"test_"+m, marker="o",
        hue="pipe", hue_order=pipe_order, palette="rocket",
        data=score_df[[c for c in score_df.columns if (m in c) or ("pipe" in c)]],
        ax=ax);

    sns.lineplot([0,1],[0,1], color="gray", ax=ax, linestyle="dotted");
    ax.set_xlim(mets[m]);
    ax.set_ylim(ax.set_xlim());
    ax.set_xlabel(m+" on training split");
    ax.set_ylabel(m+" on validation split");
    ax.legend_.remove();

ax.legend(loc="lower right", frameon=False);
fig.tight_layout();
```


    
![png](img/user-churn-benchmark/user-churn-benchmark_15_0.png)
    


We see that `mlp` & `rf` are evidently overfitted (low bias, high variance), this phenomenon is prevalent across the selected metrics.

# ROC-AUC curves

Besides, we take a peek at the overall performance, no matter what prob threshold, to identify the target class. Thus,  the plot below presents a comparison of the test roc_auc curves fitted on the first cv fold.


```python
# test set auc_roc curves on the first fold
split_get = skf.split(classification_data.loc[:,features], classification_data.target_class)
train_ind, test_ind = next(split_get)
X_, y_ = classification_data.iloc[test_ind,:].loc[:,features],\
    classification_data.target_class.iloc[test_ind]
    
roc_ls = []
for k, p in score_dict.items():
    fpr, tpr, thre = roc_curve(y_, p["estimator"][0].predict_proba(X_)[:,1])
    df = pd.DataFrame([fpr,tpr]).T.apply(pd.to_numeric)
    df["pipe"] = k
    roc_ls.append(df)
roc_df = pd.concat(roc_ls)
roc_df.columns = ["fpr","tpr","pipe"]   

fig, axs = plt.subplots(1, 2, figsize=(15,7))
sns.lineplot("fpr","tpr", hue="pipe", data=roc_df, ax=axs[0],
    palette="rocket", hue_order=pipe_order,
    estimator=None, legend=False);

sns.lineplot([0,1],[0,1], color="grey", ax=axs[0], linestyle="dotted", legend=False);

sns.lineplot("fpr","tpr", hue="pipe", data=roc_df, ax=axs[1], palette="rocket",
    hue_order=pipe_order, estimator=None,);

axs[0].set_ylabel("true positive rate");
axs[0].set_xlabel("false positive rate");
axs[1].set_ylim(0.7,0.95);
axs[1].set_xlim(0.4,0.75);     
axs[1].set_ylabel("");
axs[1].set_xlabel("false positive rate"); 
axs[1].legend(loc="lower right", frameon=False);
plt.show();
```


    
![png](img/user-churn-benchmark/user-churn-benchmark_18_0.png)
    


On the left, we can observe relationship amongst the overall roc_auc curves. The `mlp` & `rf` duo is underperforming across whole range of the fpr. On the right, we see the curves in a bit more detail.


```python
pf = open("../data/benchmark-cache.pickle","wb")
pickle.dump([classification_data, skf, score_dict], pf)
pf.close()
```

# Next steps

**Feature importance**  
 * focus on the top performing algorithms,
 * ...

**Technical**
 * factor the workhorse funcs out of the ipynb,
 * improve caching,
 * ...

> Martin Fridrich, 03/2021
