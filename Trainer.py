import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

def fetch_feature_importance(func):
    def wrapper(self, *args, **kwargs):
        # Call the original method
        result = func(self, *args, **kwargs)
        
        # Fetch feature importance if the model has the attribute
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.preprocessing:
                feature_names = self.preprocessing.get_feature_names_out()
            else:
                feature_names = [f'Feature {i}' for i in range(len(importance))]
            feature_importances_dict = dict(zip(feature_names, importance))
            self.metrics['feature_importances'] = feature_importances_dict
            print(f"-----------------------------------------------Feature Importance-----------------------------------------------")
            for feature, importance in feature_importances_dict.items():
                print(f"Feature: {feature} => Importance: {importance}")
            print(f"-----------------------------------------------The End-----------------------------------------------")

        elif hasattr(self.model, 'coef_'):
            importance = self.model.coef_
            if self.model.coef_.ndim > 1:
                importance = np.mean(np.abs(importance), axis=0)
            if self.preprocessing:
                feature_names = self.preprocessing.get_feature_names_out()
            else:
                feature_names = [f'Feature {i}' for i in range(len(importance))]
            feature_importances_dict = dict(zip(feature_names, importance))
            self.metrics['feature_importances'] = feature_importances_dict
            print(f"-----------------------------------------------Feature Importance-----------------------------------------------")
            for feature, importance in feature_importances_dict.items():
                print(f"Feature: {feature} => Importance: {importance}")
            print(f"-----------------------------------------------The End-----------------------------------------------")
        else:
            print("Model does not have feature importances or coefficients attribute.")
        
        return result
    return wrapper

class ModelTrainer:
    def __init__(self, model, preprocessing=None, param_grid=None, n_iter=10, cv=3, scoring='f1', random_state=42):
        self.model = model
        self.preprocessing = preprocessing
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.metrics = {}  # Dictionary to store metrics
        self.clf = make_pipeline(preprocessing, model) if preprocessing else model
    
    @fetch_feature_importance
    def train(self, train_X, train_y):
        print(f"Training model {self.model}")
        self.clf.fit(train_X, train_y)
        cv_score = cross_val_score(self.clf, train_X, train_y, cv=self.cv, scoring=self.scoring).mean()
        print(f"Cross-validation score: {cv_score}")
        self.metrics['cross_val_score'] = cv_score  # Store the cross-validation score
    
    def hypertune(self, train_X, train_y):
        if self.param_grid:
            print("Starting hyperparameter tuning...")
            rnd_search = RandomizedSearchCV(
                self.clf, param_distributions=self.param_grid, n_iter=self.n_iter,
                cv=self.cv, scoring=self.scoring, random_state=self.random_state
            )
            rnd_search.fit(train_X, train_y)
            self.best_estimator_ = rnd_search.best_estimator_
            print(f"Best parameters: {rnd_search.best_params_}")
            self.metrics['best_params'] = rnd_search.best_params_  # Store the best parameters
        else:
            print("No hyperparameter grid provided.")
    
    def evaluate(self, test_X, test_y):
        if hasattr(self, 'best_estimator_'):
            clf_to_use = self.best_estimator_
        else:
            clf_to_use = self.clf
        pred_y = clf_to_use.predict(test_X)
        accuracy = accuracy_score(test_y, pred_y)
        report = classification_report(test_y, pred_y, output_dict=True)
        print("Test set accuracy: ", accuracy)
        print(classification_report(test_y, pred_y))
        self.metrics['test_accuracy'] = accuracy  # Store the test accuracy
        self.metrics['classification_report'] = report  # Store the classification report
    
    def plot_roc_curve(self, test_X, test_y):
        clf_to_use = self.best_estimator_ if hasattr(self, 'best_estimator_') else self.clf
        if hasattr(clf_to_use, "decision_function"):
            y_scores = clf_to_use.decision_function(test_X)
        else:
            y_scores = clf_to_use.predict_proba(test_X)[:, 1]

        fpr, tpr, _ = roc_curve(test_y, y_scores)
        roc_auc = roc_auc_score(test_y, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: {self.model.__class__.__name__}')
        plt.legend(loc="lower right")
        plt.show()
        return plt.gcf()  # Return the current figure