import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def prepare_datasets(inut_data):
    X = inut_data.iloc[:, 1:-1]
    y = inut_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    return X_train, y_train, X_test, y_test

def tuning_hyperparamter(input_data):
    X_train, y_train, X_test, y_test=prepare_datasets(input_data)
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(2, 10)]
    min_samples_split = [2,3,4,5]
    min_samples_leaf = [1, 2,3, 4]
    bootstrap = [True, False]
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
def fit_simple_random_forest(input_data):
    X_train, y_train, X_test, y_test=prepare_datasets(input_data)
    rf_clf=RandomForestClassifier(max_depth=5,class_weight="balanced")
    return rf_clf.fit(X_train,y_train)

