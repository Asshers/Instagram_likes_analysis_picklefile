import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the CSV data
data = pd.read_csv('instagram_data3.csv')
data = data[(data['Year'] == 2020) | (data['Year'] == 2021)]

# Drop unnecessary columns
data2 = data.drop(['User uuid', 'Days passed from post', 'Numer of Comments', 'quarter', 'Minute', 'Season', 'Likes', 'Year', 'Numer of Tags', 'Day', 'timeOfDay'], axis=1)

# Define the LikeScore function
def LikeScore(n):
    if -2 <= n < -0.75:
        return 'Low Like Score'
    elif -0.75 <= n < 0.5:
        return 'Average Like Score'
    elif 0.5 <= n < 5:
        return 'High Like Score'

# Apply the LikeScore function and encode labels
data2['Likes Score'] = data2['Likes Score'].apply(LikeScore)
encoder = LabelEncoder()
data2['Likes Score'] = encoder.fit_transform(data2['Likes Score'])

data2 = data2[data2['Likes Score'] !=0 ]

# Prepare data for training
X = data2.drop('Likes Score', axis=1)
y = data2['Likes Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import expon, randint

# Define the hyperparameter search space for SVM with RBF kernel
param_dist = {
    'C': expon(scale=1),  # C is typically searched on a logarithmic scale
    'gamma': expon(scale=0.1),  # gamma is also searched on a logarithmic scale
}

# Create an SVM classifier with RBF kernel
svm_rbf = SVC(kernel='rbf')

# Perform random search
random_search_svm_rbf = RandomizedSearchCV(
    svm_rbf, param_distributions=param_dist, n_iter=10, cv=5
)

# Fit the random search to find the best hyperparameters
random_search_svm_rbf.fit(X_train, y_train)

# Print the best hyperparameters and the associated score
print("Best Hyperparameters: ", random_search_svm_rbf.best_params_)
print("Best Score: ", random_search_svm_rbf.best_score_)

best_random= random_search_svm_rbf.best_estimator_
y_pred = best_random.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Save the trained model using pickling
with open("knnclassifier.pkl", "wb") as model_file:
    pickle.dump(best_random, model_file)

unique_elements, counts = np.unique(y_pred, return_counts=True)

# Print the unique elements and their counts
for element, count in zip(unique_elements, counts):
    print(f"Element: {element}, Count: {count}")