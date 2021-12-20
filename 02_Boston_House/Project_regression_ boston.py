# Regression Project: Boston House Prices
# -----------------------------------------------------------------------------------------------
# 1. Prepare Problem
#   a) Load libraries
#   b) Load dataset
# This step is about loading everything you need to start working on your problem. This includes:
#      Python modules, classes and functions that you intend to use.
#      Loading your dataset from CSV.
# This is also the home of any global configuration you might need to do. It is also the place
# where you might need to make a reduced sample of your dataset if it is too large to work with.
# Ideally, your dataset should be small enough to build a model or create a visualization within a
# minute, ideally 30 seconds. You can always scale up well performing models later.
# Regression Project: Boston House Prices

# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)
# -----------------------------------------------------------------------------------------------
# 2. Summarize Data
# a) Descriptive statistics
# b) Data visualizations
# This step is about better understanding the data that you have available. This includes
# understanding your data using:
#   • Descriptive statistics such as summaries.
#   • Data visualizations such as plots with Matplotlib, ideally using convenience functions from Panda

# Descriptive statistics
# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head
print(dataset.head(20))
# descriptions, change precision to 2 places
set_option('precision', 1)
print(dataset.describe())
# correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))
# Data visualizations
# histograms
dataset.hist()
pyplot.show()
# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
pyplot.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# ---------------------------------------------------------------------------------------------
# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms
# This step is about preparing the data in such a way that it best exposes the structure of the
# problem and the relationships between your input attributes with the output variable. This includes tasks such as:
#   • Cleaning data by removing duplicates, marking missing values and even imputing missing values.
#   • Feature selection where redundant features may be removed and new features developed.
#   • Data transforms where attributes are scaled or redistributed in order to best expose the structure of the problem later to learning algorithms.
# --------------------------------------------------------------------------------------------
# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms
# This step is about finding a subset of machine learning algorithms that are good at exploiting the structure of your data 
# (e.g. have better than average skill). This involves steps such as:
#   • Separating out a validation dataset to use for later confirmation of the skill of your developed model.
#   • Defining test options using scikit-learn such as cross-validation and the evaluation metric to use.
#   • Spot-checking a suite of linear and nonlinear machine learning algorithms.
#   • Comparing the estimated accuracy of algorithms
# -------------------------------------------------------------------------------------------
# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles
# Once you have a shortlist of machine learning algorithms, you need to get the most out of them.
# There are two different ways to improve the accuracy of your models:
#   • Search for a combination of parameters for each algorithm using scikit-learn that yields the best results.
#   • Combine the prediction of multiple models into an ensemble prediction using ensemble techniques.
# The line between this and the previous step can blur when a project becomes concrete.
# There may be a little algorithm tuning in the previous step. And in the case of ensembles, you
# may bring more than a shortlist of algorithms forward to combine their predictions.
#---------------------------------------------------------------------------------------------
# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
# Once you have found a model that you believe can make accurate predictions on unseen data,
# you are ready to finalize it. Finalizing a model may involve sub-tasks such as:
#   • Using an optimal model tuned by scikit-learn to make predictions on unseen data.
#   • Creating a standalone model using the parameters tuned by scikit-learn.
#   • Saving an optimal model to file for later use.
#----------------------------------------------------------------------------------------------
