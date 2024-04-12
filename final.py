import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df_train=pd.read_csv('./train.csv') #read the training file
#mapping the various levels of education
education_mapping = {
    'Others': 0,
    'Literate': 1,
    '5th Pass': 2,
    '8th Pass': 3,
    '10th Pass': 4,
    '12th Pass': 5,
    'Graduate': 6,
    'Graduate Professional': 7,
    'Post Graduate': 8,
    'Doctorate': 9
}
inverse_education_map = {v: k for k, v in education_mapping.items()} #creating an inverse map
df_train['Education'] = df_train['Education'].map(education_mapping) #mapping all datapoints in the training set
# Create a mapping dictionary for party
party_mapping = {party: idx for idx, party in enumerate(df_train['Party'].unique())}
# Map the 'Party' column using the mapping dictionary
df_train['Party'] = df_train['Party'].map(party_mapping)
# Create a mapping dictionary for state
state_mapping = {state: idx for idx, state in enumerate(df_train['state'].unique())}
# Map the 'state' column using the mapping dictionary
df_train['state'] = df_train['state'].map(state_mapping)
# Custom function to check if string starts with "Dr"
def check_prefix_doc(name):
    return 1 if name.startswith('Dr.') else 0
# Custom function to check if string starts with "Adv"
def check_prefix_adv(name):
    return 1 if name.startswith('Adv.') else 0
df_train['IsDoctor']=0 #creating a new column
df_train['IsDoctor'] = df_train['Candidate'].apply(check_prefix_doc) #applying the check function to all column entries
df_train['IsAdv']=0
df_train['IsAdv'] = df_train['Candidate'].apply(check_prefix_adv)
Y_train=df_train['Education'] #storing the target variable
X_train=df_train[['Party','Criminal Case','state','IsDoctor','IsAdv']] #creating the train variable with relevant column names

#Reading the test data
df_test=pd.read_csv('./test.csv')
#similar pre-processing
df_test['Party'] = df_test['Party'].map(party_mapping)
df_test['state'] = df_test['state'].map(state_mapping)
df_test['IsDoctor']=0
df_test['IsDoctor'] = df_test['Candidate'].apply(check_prefix_doc)
df_test['IsAdv']=0
df_test['IsAdv'] = df_test['Candidate'].apply(check_prefix_adv)

#creating the test data with relevant columns
X_test=df_test[['Party','Criminal Case','state','IsDoctor','IsAdv']]
scaler = StandardScaler() #Scaling the train and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###############################   Model   ################################################################################

knn = KNeighborsClassifier() #initialising the knn classifier
# Define the parameter grid for GridSearchCV to get the accurate hyperparameters
param_grid = {
    'n_neighbors': [19,20,21],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30],
    'p': [1,2,3,4]  # p=1 for Manhattan distance, p=2 for Euclidean distance
}

# Initialize GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=10,verbose=2, scoring='f1_weighted')

# Perform grid search to find the best model
grid_search.fit(X_train, Y_train)

# Get the best model
best_knn = grid_search.best_estimator_

# Make predictions on the test data using the best model
Y_pred = best_knn.predict(X_test)

#printing the best parameters and performance on train data
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

####################################################   Making the csv file  ###############################################################

df_pred = pd.DataFrame({'Column_name': Y_pred}) #creating a dataframe
Y_pred=df_pred['Column_name']
Y_pred=Y_pred.map(inverse_education_map) #mapping the integer values back to strings
final = pd.DataFrame({'ID': range(len(Y_pred)), 'Education': Y_pred}) #preparing final submission dataframe
print(final)
final.to_csv('./submission.csv', index=False) #creating the final csv file
