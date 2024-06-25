from LoadData import Loader
import pandas as pd
import numpy as np


def main():
    # Loading the data 
    df = Loader.load(path='/workspaces/Churn_modeling/Churn_Modelling.csv')
    pd.set_option('display.max_rows', 100)  
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1500)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    print("________________________________DataFrame________________________________")
    print(df)



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
    save_plot(plt, filename = 'Relationship Between Country,Estimated Salary & Exited')

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
    import json
    preprocessing = Preprocessing()  # Instantiate the preprocessor

    def save_metrics_to_json(trainer_instance):
        """
        Save metrics from a ModelTrainer instance to a JSON file.

        Parameters:
        - trainer_instance (ModelTrainer): An instance of ModelTrainer with collected metrics.
        """
        # Generate the file path dynamically based on the model name
        metrics_file = f"/workspaces/Churn_modeling/Metrics/{trainer_instance.model}.json"

        # Fetch metrics from the trainer instance
        metrics_to_save = trainer_instance.metrics

        # Ensure the directory exists; create it if it doesn't
        save_folder = os.path.dirname(metrics_file)
        os.makedirs(save_folder, exist_ok=True)

        # Save metrics to JSON file
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_to_save, f, indent=4)
            print(f"Metrics saved to {metrics_file}")
        except IOError as e:
            print(f"Error saving metrics to {metrics_file}: {e}")


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
    save_metrics_to_json(trainer_logistic)


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
    roc_curve_rf = trainer_randomforest.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_rf, 'Rf_roc_curve')
    save_metrics_to_json(trainer_randomforest)


    # GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    param_grid = {
        'gradientboostingclassifier__learning_rate' : [0.01, 0.001, 0.0001, 0.1],
        'gradientboostingclassifier__n_estimators': randint(50, 500), 
        'gradientboostingclassifier__tol' : [0.001, 0.0001, 0.01, 0.1],
        'gradientboostingclassifier__max_features': randint(3, 13)
    }

    trainer_gradientboosting = ModelTrainer(GradientBoostingClassifier(), preprocessing, param_grid)
    trainer_gradientboosting.train(train_X, train_y)
    trainer_gradientboosting.hypertune(train_X, train_y)
    trainer_gradientboosting.evaluate(test_X, test_y)
    roc_curve_gradientboosting = trainer_gradientboosting.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_gradientboosting, 'Gradientboosting_roc_curve')
    save_metrics_to_json(trainer_gradientboosting)

    # AdaBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier

    param_grid_ada = {
        'adaboostclassifier__n_estimators': randint(50, 500),
        'adaboostclassifier__learning_rate': [0.01, 0.1, 1.0, 1.5, 2.0]
    }

    trainer_ada = ModelTrainer(AdaBoostClassifier(), preprocessing, param_grid_ada)
    trainer_ada.train(train_X, train_y)
    trainer_ada.hypertune(train_X, train_y)
    trainer_ada.evaluate(test_X, test_y)
    roc_curve_adaboosting = trainer_ada.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_adaboosting, 'AdaBoosting_roc_curve')
    save_metrics_to_json(trainer_ada)

    # Support Vector Classifier
    from sklearn.svm import SVC
    from scipy.stats import uniform

    param_grid_svc = {
        'svc__C': uniform(0.1, 10),
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svc__gamma': ['scale', 'auto']
    }

    trainer_svc = ModelTrainer(SVC(), preprocessing, param_grid_svc)
    trainer_svc.train(train_X, train_y)
    trainer_svc.hypertune(train_X, train_y)
    trainer_svc.evaluate(test_X, test_y)
    roc_curve_SVC = trainer_svc.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_SVC, 'SVC_roc_curve')
    save_metrics_to_json(trainer_svc)

    # K-Nearest Neighbors Classifier
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.stats import randint

    param_grid_knn = {
        'kneighborsclassifier__n_neighbors': randint(1, 30),
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    trainer_knn = ModelTrainer(KNeighborsClassifier(), preprocessing, param_grid_knn)
    trainer_knn.train(train_X, train_y)
    trainer_knn.hypertune(train_X, train_y)
    trainer_knn.evaluate(test_X, test_y)
    roc_curve_knn = trainer_knn.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_knn, 'Knn_roc_curve')
    save_metrics_to_json(trainer_knn)

    # ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from scipy.stats import randint

    param_grid_et = {
        'extratreesclassifier__n_estimators': randint(100, 1000),
        'extratreesclassifier__max_features': ['sqrt', 'log2', None],
        'extratreesclassifier__max_depth': [None, 10, 20, 30],
        'extratreesclassifier__min_samples_split': [2, 5, 10],
        'extratreesclassifier__min_samples_leaf': [1, 2, 4]
    }

    trainer_et = ModelTrainer(ExtraTreesClassifier(), preprocessing, param_grid_et)
    trainer_et.train(train_X, train_y)
    trainer_et.hypertune(train_X, train_y)
    trainer_et.evaluate(test_X, test_y)
    roc_curve_et = trainer_et.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_et, 'Et_roc_curve')
    save_metrics_to_json(trainer_et)

    # MLP (NN)
    from sklearn.neural_network import MLPClassifier
    from scipy.stats import uniform

    param_grid_mlp = {
        'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
        'mlpclassifier__alpha': uniform(0.0001, 0.01),
        'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'mlpclassifier__max_iter': randint(1000, 2500), 
        'mlpclassifier__early_stopping': [True, False]
    }

    trainer_mlp = ModelTrainer(MLPClassifier(max_iter=1000), preprocessing, param_grid_mlp)
    trainer_mlp.train(train_X, train_y)
    trainer_mlp.hypertune(train_X, train_y)
    trainer_mlp.evaluate(test_X, test_y)
    roc_curve_MLP = trainer_mlp.plot_roc_curve(test_X, test_y)
    save_plot(roc_curve_MLP, 'MLP_roc_curve')
    save_metrics_to_json(trainer_mlp)
if __name__ == "__main__":
    main()