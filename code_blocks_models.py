############################################### Models tried ##################################################

#1. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier #use this import statement at the top
# Initialize Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
# Define hyperparameters to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Initialize GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='f1_weighted')
# Perform grid search to find the best model
grid_search.fit(X_train, Y_train)
# Get the best model
best_rf_clf = grid_search.best_estimator_
# Make predictions on the test data using the best model
y_pred = best_rf_clf.predict(X_test)
# Print out the best parameters
print("Best Parameters:", grid_search.best_params_)

#####################################################################################################
#2. Support Vector Classifier

from sklearn.svm import SVC #use this import statement on the top
svc_clf = SVC(random_state=42)
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
grid_search_svc = GridSearchCV(svc_clf, param_grid_svc, cv=5, scoring='f1_weighted')
grid_search_svc.fit(X_train, Y_train)
best_svc_clf = grid_search_svc.best_estimator_
y_pred_svc = best_svc_clf.predict(X_test)
print("SVC Best Parameters:", grid_search_svc.best_params_)

########################################################################################################
#3. Decision trees classifier

from sklearn.tree import DecisionTreeClassifier #use this import statement on the top
dt_clf = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(dt_clf, param_grid_dt, cv=5, scoring='f1_weighted')
grid_search_dt.fit(X_train, Y_train)
best_dt_clf = grid_search_dt.best_estimator_
y_pred_dt = best_dt_clf.predict(X_test)
print("Decision Tree Best Parameters:", grid_search_dt.best_params_)

########################################################################################################
