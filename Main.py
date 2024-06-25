from LoadData import Loader
import pandas as pd
import numpy as np
# Loading the data 
df = Loader.load(path='/workspaces/Churn_modeling/Churn_Modelling.csv')


print("________________________________DataFrame________________________________")
print(df.head())



print("________________________________DataFrame Information________________________________")
print(df.info())



print("________________________________Stats & Info of Columns________________________________")
print(df.describe())


"""____________________________________________________________________________________"""

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Save Figures
def save_plot(plt, filename : str, save_folder = '/workspaces/Churn_modeling/Plots') -> None:
    """

    This Function is for saving the figures/Plots/Graphs

    plt = plot you want to save

    default path to save the figures = '/workspaces/Churn_modeling/Plots'

    filename = Name of the figure 

    """

    try:
        os.makedirs(save_folder, exist_ok=True)
        print(f"Directory '{save_folder}' created successfully or already exists.")
    except OSError as e:
        print(f"Error: Creating directory '{save_folder}'. {e}")
    
    plot_filename = os.path.join(save_folder, filename)
    
    # Save the plot
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved successfully to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")



# Distribution of Data
numeric_columns = df.select_dtypes(include='number').columns

num_plots = len(numeric_columns)

# grid size
n_cols = 3
n_rows = (num_plots + n_cols - 1) // n_cols

# subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

axes = axes.flatten()
sns.set(style="whitegrid")
# Plot each column in a separate subplot
for ax, column in zip(axes, numeric_columns):
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# Remove any empty subplots
for ax in axes[num_plots:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.show()
save_plot(plt, 'Distribution of Dataset')


# Geography vs Exited Plot
plt.figure(figsize=(5, 3))
sns.countplot(x='Geography', data=df, hue='Exited', palette ='Set2')
plt.title('Geography wise (Exited 1:Yes, 0:No)')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()
save_plot(plt, 'Geography vs Exited')

# Pair plot
sns.pairplot(df, hue='Exited', palette ='Set2')
plt.suptitle('Histograms of Various Features', y=1.02)
plt.show()
save_plot(plt, 'PairPlots')

# Relationship B/W Country X Estimated Salary & Exited
sns.catplot(data=df,x='Geography', y='EstimatedSalary', hue='Exited', kind='box', height=4, palette ='Set2')
plt.title('Relationship B/W Country X Estimated Salary & Exited (Exited 1:Yes, 0:No)')
save_plot(plt, 'Relationship B/W Country X Estimated Salary & Exited')

# Correlation Matrix
plt.figure(figsize=(10, 7))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
save_plot(plt, 'Correlation Matrix')

# Train Test Split
from sklearn.model_selection import train_test_split, cross_val_score
features = df.drop('Exited', axis=1).copy()
labels = df['Exited'].copy()

train_X, test_X, train_y, test_y = train_test_split(features, labels, stratify=df['Exited'] ,
                                                    train_size=0.80, shuffle=True, random_state=42)
print(f"""
      train_X Size : {len(train_X)}
      train_y : {len(train_y)}
      test_X : {len(test_X)}
      test_y : {len(test_y)}
""")

# Preprocessing and Model Training

from Trainer import ModelTrainer
from Preprocessor import Preprocessing
from scipy.stats import randint

preprocessing = Preprocessing()  # Instantiate the preprocessor

# Logistic Regression
from sklearn.linear_model import LogisticRegression

param_grid = {
    'logisticregression__tol': [1e-2, 1e-3, 1e-4, 1e-5],
    'logisticregression__C': randint(1, 20),
    'logisticregression__max_iter': [100, 150, 200, 250]
}

trainer_logistic = ModelTrainer(LogisticRegression(random_state=42), preprocessing, param_grid)
trainer_logistic.train(train_X, train_y)
trainer_logistic.hypertune(train_X, train_y)
trainer_logistic.evaluate(test_X, test_y)
roc_curve_logistic = trainer_logistic.plot_roc_curve(test_X, test_y)
save_plot(roc_curve_logistic, 'Logistic_roc_curve')

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'randomforestclassifier__n_estimators': randint(100, 500),  # Adjusted range
    'randomforestclassifier__max_features': [None, 'sqrt', 'log2', 0.5],
    'randomforestclassifier__max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'randomforestclassifier__min_samples_split': randint(2, 20),
    'randomforestclassifier__min_samples_leaf': randint(1, 20),
    'randomforestclassifier__bootstrap': [True, False]
}

trainer_randomforest = ModelTrainer(RandomForestClassifier(), preprocessing, param_grid)
trainer_randomforest.train(train_X, train_y)
trainer_randomforest.hypertune(train_X, train_y)
trainer_randomforest.evaluate(test_X, test_y)
trainer_randomforest.plot_roc_curve(test_X, test_y)
save_plot(roc_curve_logistic, 'rf_roc_curve')