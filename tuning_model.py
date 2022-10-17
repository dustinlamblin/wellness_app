from sklearn.model_selection import RandomizedSearchCV, train_test_split


def prepare_datasets(input_data):
    X = input_data.iloc[:,:-1]
    y = input_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, shuffle=True
    )
    return X_train, y_train, X_test, y_test


