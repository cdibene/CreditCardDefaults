#Cameron DiBenedetto
#Machine Learning in Credit Card Defaults
#April 2nd 2020

# here we will import the libraries used for machine learning
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import randint
import pandas as pd  # data processing, CSV file I/O, data manipulation
import matplotlib.pyplot as plt  # this is used for the plot the graph
import seaborn as sns  # used for plot interactive graph.
from pandas import set_option

plt.style.use('ggplot')  # nice plots

from sklearn.model_selection import train_test_split  # to split the data into two parts
from sklearn.linear_model import LogisticRegression  # to apply the Logistic regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold  # for cross validation
from sklearn.model_selection import GridSearchCV  # for tuning parameter
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.preprocessing import StandardScaler  # for normalization
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from xgboost import XGBClassifier

from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics  # for the check the error and accuracy of the model
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import Image
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

import os

# File path is different for anyone find where you put the file
data = pd.read_excel('/Users/mrcajuste/Desktop/default_of_credit_card_clients.xls')
data.sample(5)
# print("DATA")
# print(data)
# print("test")
#
data.rename(columns={"default.payment.next.month": "Default"}, inplace=True)
data.drop('ID', axis=1, inplace=True)  # drop column "ID"
data.info()

print(data.columns)

# Separating features and target
y = data.Default  # target default=1 or non-default=0
features = data.drop('Default', axis=1, inplace=False)

data['EDUCATION'].unique()
# array([2, 1, 3, 5, 4, 6, 0])

data['EDUCATION'] = np.where(data['EDUCATION'] == 5, 4, data['EDUCATION'])
data['EDUCATION'] = np.where(data['EDUCATION'] == 6, 4, data['EDUCATION'])
data['EDUCATION'] = np.where(data['EDUCATION'] == 0, 4, data['EDUCATION'])

data['EDUCATION'].unique()
# array([2, 1, 3, 4])
data['MARRIAGE'].unique()
# array([1, 2, 3, 0])
data['MARRIAGE'] = np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])
data['MARRIAGE'].unique()
# array([1, 2, 3])

# print(data.info)
#
# data.head(10)
yes = data.Default.sum()
no = len(data) - yes

# Percentage
yes_perc = round(yes / len(data) * 100, 1)
no_perc = round(no / len(data) * 100, 1)

import sys

plt.figure(figsize=(7, 4))
sns.set_context('notebook', font_scale=1.2)
sns.countplot('Default', data=data, palette="Blues")
plt.annotate('Non-default: {}'.format(no), xy=(-0.3, 15000), xytext=(-0.3, 3000), size=12)
plt.annotate('Default: {}'.format(yes), xy=(0.7, 15000), xytext=(0.7, 3000), size=12)
plt.annotate(str(no_perc) + " %", xy=(-0.3, 15000), xytext=(-0.1, 8000), size=12)
plt.annotate(str(yes_perc) + " %", xy=(0.7, 15000), xytext=(0.9, 8000), size=12)
plt.title('COUNT OF CREDIT CARDS', size=14)
# Removing the frame
plt.box(False);

# plt.show()
#
# print("done1")

# Creating a new dataframe with categorical variables
subset = data[['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4',
               'PAY_5', 'PAY_6', 'Default']]

f, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor='white')
f.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')
ax1 = sns.countplot(x="SEX", hue="Default", data=subset, palette="Blues", ax=axes[0, 0])
ax2 = sns.countplot(x="EDUCATION", hue="Default", data=subset, palette="Blues", ax=axes[0, 1])
ax3 = sns.countplot(x="MARRIAGE", hue="Default", data=subset, palette="Blues", ax=axes[0, 2])
ax4 = sns.countplot(x="PAY_0", hue="Default", data=subset, palette="Blues", ax=axes[1, 0])
ax5 = sns.countplot(x="PAY_2", hue="Default", data=subset, palette="Blues", ax=axes[1, 1])
ax6 = sns.countplot(x="PAY_3", hue="Default", data=subset, palette="Blues", ax=axes[1, 2])
ax7 = sns.countplot(x="PAY_4", hue="Default", data=subset, palette="Blues", ax=axes[2, 0])
ax8 = sns.countplot(x="PAY_5", hue="Default", data=subset, palette="Blues", ax=axes[2, 1])
ax9 = sns.countplot(x="PAY_6", hue="Default", data=subset, palette="Blues", ax=axes[2, 2]);

# plt.show()
print("done2")

stdX = (features - features.mean()) / (features.std())  # standardization
# dataset
X = data.drop('Default', axis=1)  # rest of the data
y = data['Default']  # targer data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# cModel = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=6)
#
# clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
#                               max_features=None, max_leaf_nodes=6, min_impurity_decrease=0.0,
#                               min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
#                               min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
#
#
# #clf = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=6)
# cModel = clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
#
# dot_data = StringIO()
# tree.export_graphviz(cModel, out_file=dot_data, filled=True, rounded=True, special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# print("right before create")
# # Image(graph.create_png())
# print("Accuracy Score")
# print(acc)
# print("right after create")
# graph.write_pdf("pruned graph.pdf")
#
# dtree = DecisionTreeClassifier(criterion='gini')
# dtree.fit(X_train, y_train)
# pred = dtree.predict(X_test)
# print("Accuracy Results:")
# print('Criterion = gini:', accuracy_score(y_test, pred))
# dtree = DecisionTreeClassifier(criterion='entropy')
# dtree.fit(X_train, y_train)
# pred = dtree.predict(X_test)
# print('Criterion = entropy', accuracy_score(y_test, pred))
#
# # not sure effects anything but it supposed to make the graph
# max_depth = []
# acc_gini = []
# acc_entropy = []
# for i in range(1, 30):
#     dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
# dtree.fit(X_train, y_train)
# pred = dtree.predict(X_test)
# acc_gini.append(accuracy_score(y_test, pred))
# ####
# dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
# dtree.fit(X_train, y_train)
# pred = dtree.predict(X_test)
# acc_entropy.append(accuracy_score(y_test, pred))
# ####
# max_depth.append(i)
# d = pd.DataFrame({'acc_gini': pd.Series(acc_gini), 'acc_entropy': pd.Series(acc_entropy),
#                   'max_depth': pd.Series(max_depth)})
# # visualizing changes in parameters
# plt.plot('max_depth', 'acc_gini', data=d, label='gini')
# plt.plot('max_depth', 'acc_entropy', data=d, label='entropy')
# plt.xlabel('max_depth')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()


# regular graph

# print("checkpoint2")
# dtree = DecisionTreeClassifier(criterion='entropy', max_depth=6, splitter='random')
# pModel = dtree.fit(X_train, y_train)
# pred = dtree.predict(X_test)
# ascore = accuracy_score(y_test, pred)
# print(ascore)
#
# dot_data2 = StringIO()
# tree.export_graphviz(pModel, out_file=dot_data2, filled=True, rounded=True, special_characters=True)
# pgraph = pydotplus.graph_from_dot_data(dot_data2.getvalue())
# pgraph.write_pdf("Graph.pdf")
#
# print("right after create")

# Dataset with standardized features
Xstd_train, Xstd_test, ystd_train, ystd_test = train_test_split(stdX, y, test_size=0.2, stratify=y)

# Dataset with three most important features
Ximp = stdX[['PAY_0', 'BILL_AMT1', 'PAY_AMT2']]
X_tr, X_t, y_tr, y_t = train_test_split(Ximp,y, test_size=0.2, stratify=y, random_state=42)



                                                                #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

print(y.shape)

LR = LogisticRegression(C=0.00005, random_state=0)
print(y.shape)
LR.fit(X_train, y_train)
print(y.shape)
y_pred = LR.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation
cv_scores =cross_val_score(LR, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_test,LR.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Logistic Regression");
plt.show()


LRS = LogisticRegression(C=3.73, random_state=0)
LRS.fit(Xstd_train, ystd_train)
y_pred = LRS.predict(Xstd_test)
print('Accuracy:', metrics.accuracy_score(y_pred,ystd_test))

## 5-fold cross-validation
cv_scores =cross_val_score(LRS, stdX, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(ystd_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(ystd_test,LRS.predict(Xstd_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Logistic Regression with standardized data");
plt.show


LR_imp = LogisticRegression(C=3.73, random_state=0)
LR_imp.fit(X_tr, y_tr)
y_pred = LR_imp.predict(X_t)
print('Accuracy:', metrics.accuracy_score(y_pred,y_t))

## 5-fold cross-validation
cv_scores =cross_val_score(LR_imp, Ximp, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_t, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_t,LR_imp.predict(X_t))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Logistic Regression (most important features)");
print("finish")

Tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 7,
                                     max_features= 9, min_samples_leaf= 2,
                                     random_state=0)
Tree.fit(X_train, y_train)
y_pred = Tree.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))
plt.show()

## 5-fold cross-validation
cv_scores =cross_val_score(Tree, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_test,Tree.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Decision Tree");
plt.show()
Ran = RandomForestClassifier(criterion= 'gini', max_depth= 6,
                                     max_features= 5, n_estimators= 150,
                                     random_state=0)
Ran.fit(X_train, y_train)
y_pred = Ran.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation
cv_scores =cross_val_score(Ran, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Random Forest");
plt.show()

#ROC and AUC
y_pred_proba_RF = Ran.predict_proba(X_test)[::,1]
fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba_RF)
auc1 = metrics.roc_auc_score(y_test, y_pred_proba_RF)

y_pred_proba_DT = Tree.predict_proba(X_test)[::,1]
fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba_DT)
auc2 = metrics.roc_auc_score(y_test, y_pred_proba_DT)

y_pred_proba_LR = LR.predict_proba(X_test)[::,1]
fpr3, tpr3, _ = metrics.roc_curve(y_test,  y_pred_proba_LR)
auc3 = metrics.roc_auc_score(y_test, y_pred_proba_LR)

y_pred_proba_LRS = LRS.predict_proba(Xstd_test)[::,1]
fpr4, tpr4, _ = metrics.roc_curve(ystd_test,  y_pred_proba_LRS)
auc4 = metrics.roc_auc_score(ystd_test, y_pred_proba_LRS)

y_pred_proba_LRimp = LR_imp.predict_proba(X_t)[::,1]
fpr5, tpr5, _ = metrics.roc_curve(y_t,  y_pred_proba_LRimp)
auc5 = metrics.roc_auc_score(y_t, y_pred_proba_LRimp)

plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="Random Forest, auc="+str(round(auc1,2)))
plt.plot(fpr2,tpr2,label="Decision Tree, auc="+str(round(auc2,2)))
plt.plot(fpr3,tpr3,label="LogReg, auc="+str(round(auc3,2)))
plt.plot(fpr4,tpr4,label="LogReg(std), auc="+str(round(auc4,2)))
plt.plot(fpr5,tpr5,label="LogReg(Std&Imp), auc="+str(round(auc5,2)))
plt.legend(loc=4, title='Models', facecolor='white')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC', size=15)
plt.box(False)
plt.savefig('ImageName', format='png', dpi=200, transparent=True);
plt.show()
