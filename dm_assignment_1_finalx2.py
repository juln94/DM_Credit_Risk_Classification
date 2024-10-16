# **Import Modules**

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os

# set working directory
# modify the path consists of the data file as required 
path = "/Users/thunguyen/Library/CloudStorage/OneDrive-LATROBEUNIVERSITY/CSE5DMI/CSE5DMI - Assignment 1"
os.chdir(path)

"""# **DM - Assignment 1**

## 0. Utilities
"""

# Dataset overview
def data_overview(df, label_col_name):
  print("\nSamples from head:\n", df.head())
  print("\nSamples from tail:\n", df.tail())
  print("\nStats of numeric columns:\n", df.describe())
  print("\nData types:\n")
  print(df.info())
  print("\nDimensions (rows, columns):\n", df.shape)
  for col in df.columns:
    print("\nUnique values in", col)
    print(dataset[col].unique())

  print("Number of records by label:")
  print(dataset[label_col_name].value_counts(dropna=False)) #NaN will be included in the count

# Identify missing values
def count_missing_val(df, label_col_name):
  # Replace infinite values with NaN
  df = df.replace([np.inf, -np.inf], np.nan)

  # Count missing values which belong to bad label
  is_bad = []
  for col in df.columns:
    is_bad.append(
        (
            ((col != label_col_name) & (df[col].isnull())) & # Exclude 'Default' col
            (df[label_col_name] == 1) # Check if row label is bad
        ).sum()
      )
  # Identify number of missing values in each column
  missing_value_counts = pd.DataFrame({
    'num_missing_vals': df.isnull().sum(),
    '%_missing': df.isnull().sum() / len(df) * 100,
    'is_bad_count': is_bad
  })
  print(missing_value_counts.sort_values('num_missing_vals', ascending=False))

  # Count total missing values per row
  missing_counts = df.isnull().sum(axis=1)
  print("Total row with missing value(s):", (missing_counts >= 1).sum())
  print("Rows with 1 missing value:", (missing_counts == 1).sum())
  print("Rows with >=2 missing values:", (missing_counts >= 2).sum())

#Converts float columns in a DataFrame to int if all values are integers
def data_type_preprocess(df):
  for col in df.columns:
    # auto type float64
    if df[col].dtype == 'float64':
      # Check if all non-NaN values in the column are integer
      if df[col].dropna().apply(lambda x: x.is_integer()).all():
        # Then convert that column to int
        df[col] = df[col].astype(int, errors='ignore')
  return df

"""## 1. Explore, aggregate and transform the attributes

### a. Load + explore
"""

# Load csv dataset file as dataframe + explore
dataset = pd.read_csv("Assignment 1 Data.csv", header=0)
data_overview(dataset, 'Default')

# Identify missing values
count_missing_val(dataset, 'Default')

"""### b. Clean"""

# Handle columns with missing val
dataset['duration'].fillna(int(dataset['duration'].mean()), inplace=True) # Replace NaN with mean age
dataset['purpose'].fillna('A410', inplace=True) # A410: Others
dataset['savings'].fillna('A65', inplace=True) # A65: unknown/ no savings account
dataset['others'].fillna('A101', inplace=True) # Other debtors / guarantors A101: none
dataset['amount'].fillna(0, inplace=True)
dataset['age'].fillna(int(dataset['age'].mean()), inplace=True) # Replace NaN with mean age
dataset['checkingstatus1'].fillna('A14', inplace=True) # A14: no checking account
dataset['property'].fillna('A124', inplace=True) # A124: unknown / no property

job_mask = dataset['job'].isnull() # boolean mask indicating which rows in the 'job' column have NaN values
employ_mask = dataset['employ'] == 'A71' # boolean mask indicating which rows in the 'employ' column have unemployed (A71) values
dataset.loc[job_mask & employ_mask, 'job'] = 'A171' # Select the rows where both conditions are true --> A171: unemployed/ unskilled - non-resident

# Drop rows with missing values in ['duration', 'purpose', 'job', 'housing', 'foreign']
drop_cols = ['job', 'housing', 'foreign']
dataset = dataset.dropna(subset=drop_cols)

# Converts float columns in a DataFrame to int if all values are integers
dataset = data_type_preprocess(dataset)

# Encode categorical column
# https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
ordinal_cols = ['checkingstatus1', 'history', 'savings', 'employ', 'others', 'property', 'otherplans', 'housing', 'job', 'foreign']
onehot_cols = [col for col in categorical_cols if col not in ordinal_cols]

# Preprocess ordinal_cols
# Higher order = Higher risk
# 0 = unknown
dataset['checkingstatus1'].replace({'A11': 3, 'A12': 2, 'A13': 1, 'A14': 0}, inplace=True) #A14: no checking account
dataset['history'].replace({'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}, inplace=True) #A30: no credits; A34: critical
dataset['savings'].replace({'A61': 4, 'A62': 3, 'A63': 2, 'A64': 1, 'A65': 0}, inplace=True) #A65: unknown/ no savings account
dataset['employ'].replace({'A71': 5, 'A72': 4, 'A73': 3, 'A74': 2, 'A75': 1}, inplace=True) #A71: unemployed
dataset['others'].replace({'A101': 3, 'A102': 2, 'A103': 1}, inplace=True) # A101 : none
dataset['property'].replace({'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4}, inplace=True) # real estate --> none
dataset['otherplans'].replace({'A141': 2, 'A142': 1, 'A143': 0}, inplace=True) # A153: none
dataset['housing'].replace({'A151': 2, 'A152': 1, 'A153': 0}, inplace=True) # A153: for free ???
dataset['job'].replace({'A171': 4, 'A172': 3, 'A173': 2, 'A174': 1}, inplace=True) # A171 : unemployed
dataset['foreign'].replace({'A201': 2, 'A202': 1}, inplace=True) # A201 : yes = non-resident
# Preprocess one-hot encoding cols
dataset = pd.get_dummies(dataset, columns=onehot_cols)

onehot_cols

# Review
#data_overview(dataset, 'Default')

"""### c. Export Cleaned Dataset"""

dataset.to_csv("data_clean.csv", encoding='utf-8', index=False, header=True)

"""## 2. Classification

### a. Load the cleaned dataset
"""

# Load exported csv dataset file as dataframe + recheck
dataset_clean = pd.read_csv("data_clean.csv", header=0)
#data_overview(dataset_clean, 'Default')

"""### b.Building the learner, and performing 10-fold cross-validation"""

# Split dataset into features set X and target variable y
X = dataset_clean.drop('Default', axis=1)  # Replace 'target_variable' with your actual target column
y = dataset_clean['Default']

# Create instance of decision tree classifier
classifier = DecisionTreeClassifier(random_state=310)

# Perform 10-fold cross-validation
scores = cross_val_score(classifier, X, y, cv=10, scoring='accuracy')
print("Avg accuracy (k-fold):", scores.mean())

# Perform on the entire dataset
# Train/test dataset split
# Set fixed random_state for guarantee getting the same result every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=310)
# Train
classifier.fit(X_train, y_train)
# Make predictions
y_pred = classifier.predict(X_test)

"""### c. Evaluation"""

def plot_roc(fp_rate, tp_rate, roc_auc):
  plt.figure(figsize=(8, 6))
  plt.plot(fp_rate, tp_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0,1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.show()

def eval_metrics(y_test, y_pred):
  # Confusion matrix
  # 0: Good (Negative)
  # 1: Bad (Positive)
  cm = confusion_matrix(y_test, y_pred)
  print("Confusion matrix:\n", cm)

  # Accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)

  # ROC curve and AUC
  fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(fp_rate, tp_rate)
  plot_roc(fp_rate, tp_rate, roc_auc)
  print("AUC:", roc_auc)

  # Cost Matrix
  # 10: FN - class a record as good when they are bad
  # 1: FP - class a customer as bad when they are good
  cost_matrix = [[0, 1],
                 [10, 0]]
  cost = (cm[0, 1] * cost_matrix[0][1]) + (cm[1, 0] * cost_matrix[1][0])
  print("Cost:", cost)

def eval_metrics_2(y_test, y_pred):
  # Accuracy
  accuracy = accuracy_score(y_test, y_pred)

  # ROC curve and AUC
  fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(fp_rate, tp_rate)

  # Cost Matrix
  cm = confusion_matrix(y_test, y_pred)
  cost_matrix = [[0, 1], [10, 0]]
  cost = (cm[0, 1] * cost_matrix[0][1]) + (cm[1, 0] * cost_matrix[1][0])
  #print("Accuracy:", accuracy, "AUC:", roc_auc, "Cost:", cost)
  return {"acc": accuracy, "roc_auc": roc_auc, "cost": cost, "cm": cm}

eval_metrics(y_test, y_pred)

"""Cost too high
- Lots of FN
- Imbalance ratio (good to bad) 7:3

### d. Fine tune using cost matrix

https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/
"""

def grid_search_decision_tree(X_train, X_test, y_train, y_test, cost_matrix, params_grid):
  results = []

  for criteria in params_grid['criterion']:
    for md in params_grid['max_depth']:
      for mss in params_grid['min_samples_split']:
        for msl in params_grid['min_samples_leaf']:
          for mf in params_grid['max_features']:
            for mwfl in params_grid['min_weight_fraction_leaf']:
              for class_weight in params_grid['class_weight']:
                if md == 0:
                  md = None

                # Config classifier
                classifier = DecisionTreeClassifier(
                  criterion=criteria,
                  max_depth=md,
                  min_samples_split=mss,
                  min_samples_leaf=msl,
                  max_features=mf,
                  min_weight_fraction_leaf=mwfl/10,
                  class_weight=class_weight,
                  random_state=310
                )

                # Train
                classifier.fit(X_train, y_train)

                # Make predictions
                y_train_pred = classifier.predict(X_train)
                y_pred = classifier.predict(X_test)

                result = {
                    'train': eval_metrics_2(y_train, y_train_pred),
                    'test': eval_metrics_2(y_test, y_pred),
                    'params':{
                      'criterion': criteria,
                      'max_depth': md,
                      'min_samples_split': mss,
                      'min_samples_leaf': msl,
                      'max_features': mf,
                      'min_weight_fraction_leaf': mwfl/10,
                      'class_weight': class_weight,
                      'random_state': 310
                    }
                }

                results.append(result)

  return results

# Define parameter ranges
params_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
          'max_depth': np.arange(3, 11),
          'min_samples_split': np.arange(2, 11),
          'min_samples_leaf': np.arange(1, 6),
          'min_weight_fraction_leaf': np.arange(0, 6)/10,
          'class_weight': ['balanced', None],
          'max_features': ['sqrt', 'log2', None]
        }
cost_matrix = [[0, 1], [10, 0]]

# Run grid search
results = grid_search_decision_tree(X_train, X_test, y_train, y_test, cost_matrix, params_grid)

def find_local_minima(test_costs):
    test_costs_np = np.array(test_costs)

    # Find the change sign points
    diff_signs = np.diff(np.sign(np.diff(test_costs_np)))
    local_minima_indices = np.where(diff_signs == 2)[0] + 1

    return local_minima_indices

def filter_local_minima(sorted_data):
    # Extract test costs
    test_costs = [item['test']['cost'] for item in sorted_data]

    # Find the indices of local minima
    local_minima_indices = find_local_minima(test_costs)

    # Filter the sorted data to keep only result with test cost at local minima
    filtered_data = [sorted_data[i] for i in local_minima_indices]

    return filtered_data

def sort_performance(results):
    # Sort the data by 'train' cost in descending order, then by 'train' acc in descending order
    # The minus sign before x['train']['cost'] negates these value --> descending
    sorted_results = sorted(results, key=lambda x: (-x['train']['cost'], x['train']['acc']))
    return sorted_results


def plot_performance(results, threshold_train=0, threshold_test=0):
    # Filter data with cost > threshold
    filtered_data = results
    if (threshold_train != 0):
      filtered_data = [item for item in filtered_data if item['train']['cost'] <= threshold_train]
    if (threshold_test != 0):
      filtered_data = [item for item in filtered_data if item['test']['cost'] <= threshold_test]

    # Extract 'train' and 'test' costs for plotting
    train_costs = [item['train']['cost'] for item in filtered_data]
    test_costs = [item['test']['cost'] for item in filtered_data]

    # Plot the sorted data
    plt.plot(range(len(filtered_data)), train_costs, label='Train Cost')
    plt.plot(range(len(filtered_data)), test_costs, label='Test Cost')
    plt.xlabel('Model ID')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

"""#### All models' results sorted"""

print('All results (sorted):')
results_sorted = sort_performance(results)
plot_performance(results_sorted)

print('Sorted results (cost train <= 400):')
plot_performance(results_sorted, 400)

# Return result with lowest cost and highest acc (train)
print('Best train params:\n', results_sorted[-1])

best_classifier = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', min_weight_fraction_leaf=0.0, class_weight = 'balanced', random_state=310)
best_classifier.fit(X_train, y_train)
text_represent_best = tree.export_text(best_classifier, feature_names=list(X_train.columns))
print(text_represent_best)

y_train_pred = best_classifier.predict(X_train)
eval_metrics(y_train, y_train_pred)

y_pred = best_classifier.predict(X_test)
eval_metrics(y_test, y_pred)

"""#### Filter only local minima of test cost"""

print('All results (sorted, keep only local minima of test cost):')
filtered_results = filter_local_minima(results_sorted)
plot_performance(filtered_results)

print('Filtered results (cost train <= 400):')
plot_performance(filtered_results, 400)

# Return result with lowest cost and highest acc (train)
print('Best train params (filtered):\n', filtered_results[-1])

best_classifier = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=2, min_samples_leaf=1, max_features=None, min_weight_fraction_leaf=0.0, class_weight = 'balanced', random_state=310)
best_classifier.fit(X_train, y_train)
text_represent_best = tree.export_text(best_classifier, feature_names=list(X_train.columns))
print(text_represent_best)

y_train_pred = best_classifier.predict(X_train)
eval_metrics(y_train, y_train_pred)

y_pred = best_classifier.predict(X_test)
eval_metrics(y_test, y_pred)