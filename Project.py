# Rishi Siddharth
# Sophomore at American University Studying Data Science and International Business

import numpy as np
import pandas as pd
import requests
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming the script is run from the user's home directory
df_new = pd.read_csv("~/Desktop/Itec 400/Datasets/apartments.csv")
columns_to_convert = ['price', 'square_feet', 'bathrooms', 'bedrooms', 'latitude', 'longitude']
for column in columns_to_convert:
    df_new[column] = pd.to_numeric(df_new[column], errors='coerce')

df_new.dropna(subset=columns_to_convert, inplace=True)

df_new.dropna(subset=['state'], inplace=True)

df_new = df_new[df_new['state'].str.match('^[A-Z]{2}$') == True]

df_new.reset_index(drop=True, inplace=True)

df_new

# Part 2 - Multivariate Clustering of Apartment Listings


# Part 2 - Starter Task

apartment_listing = KMeans(n_clusters=51, random_state=12345)
apartment_listing.fit(df_new[['latitude', 'longitude']])

# Predict the clusters
new_kmeans = apartment_listing.predict(df_new[['latitude', 'longitude']])

# Plot the clustering result with a wider figure size
plt.figure(figsize=(10, 6))  # Adjusted for wider plot
plt.scatter(df_new['longitude'], df_new['latitude'], c=new_kmeans, cmap='viridis', s=10)

# Plot the cluster centers
centers = apartment_listing.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='red', s=50, marker='x')

# Set plot labels and title
plt.title('KMeans Clustering of Apartments (51 Clusters)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')

# Display the plot
plt.show()


# Part 2 - Enhanced Task Code:
#creating a new list
#Enhanced Task
#Augmented my analysis

apartment_listing = KMeans(n_clusters=67, random_state=12345)
apartment_listing.fit(df_new[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms']])

# Predict the clusters
clusters = apartment_listing.predict(df_new[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms']])

# Plotting the 2D scatter plot for latitude and longitude
plt.figure(figsize=(10, 6))
plt.scatter(df_new['longitude'], df_new['latitude'], c=clusters, cmap='viridis', s=10)

# Plotting the cluster centers based on latitude and longitude only
centers = apartment_listing.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='red', s=50, marker='x')  # centroids in lat-long space

plt.title('KMeans Clustering of Apartments (67 Clusters)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster Label')
plt.show()



# Part 3 - Starter Task Code:
       #linear regression model to predict apartment rental prices.
#1 Data Preparation
pd.set_option('display.float_format', lambda x: '%.3f' % x) #converst to decimal format
df_sum = df_new[['bathrooms', 'bedrooms', 'square_feet']]

# Summing only the features
df_sum = df_new[['bathrooms', 'bedrooms', 'square_feet']].mean()
print(df_sum)
#means 
#bathrooms        1.445
#bedrooms         1.728
#square_feet    955.759
#price         1525.871

#1b splitting data
features = ['bathrooms', 'bedrooms', 'square_feet']
target = 'price'
df_new = df_new[features + [target]]  # Ensure df is your original DataFrame

X = df_new[features]
Y = df_new[target]
Xvar, X_test, yvar, y_test = train_test_split(X, Y, test_size=0.10, random_state=12345)

#2 - Baseline Prediction
# Calculate the mean price from the training set
mean_price_train = yvar.mean()


baseline_predictions = [mean_price_train] * len(y_test)
rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

print("Mean Price in Training Set:", mean_price_train)

#The mean of the subset of the group

#3 - Linear Regression Model 
#intiliazing the linear Model

linear_model = LinearRegression()
linear_model.fit(Xvar, yvar)  # Train on the actual training data

# Perform 10-fold cross-validation
rmse_scorer = make_scorer(mean_squared_error, squared=False)
cv_scores = cross_val_score(linear_model, X, Y, cv=10, scoring=rmse_scorer)
average_rmse = cv_scores.mean()

# 4 - Model Training and Evaluation 
y_pred = linear_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

# 5 - Output
print(f"Average RMSE from cross-validation: {average_rmse}")
print(f"RMSE on test set: {rmse_test}")
print(f"Baseline RMSE using average price: {rmse}")

#output results
#-----------------------
#bathrooms       1.445
#bedrooms        1.728
#square_feet   955.759
#dtype: float64
#Mean Price in Training Set: 1526.3446878234263
#Average RMSE from cross-validation: 805.1537806020149
#RMSE on test set: 766.6601706141287
#Baseline RMSE using average price: 847.906170646509



#resetting the data frame
df_new = pd.read_csv("~/Desktop/Itec 400/Datasets/apartments.csv")
columns_to_convert = ['price', 'square_feet', 'bathrooms', 'bedrooms', 'latitude', 'longitude']
for column in columns_to_convert:
    df_new[column] = pd.to_numeric(df_new[column], errors='coerce')

df_new.dropna(subset=columns_to_convert, inplace=True)

df_new.dropna(subset=['state'], inplace=True)

df_new = df_new[df_new['state'].str.match('^[A-Z]{2}$') == True]


df_new.reset_index(drop=True, inplace=True)


df_new


#Enhancing my linear regression model by experimenting with different feature
#engineering strategies to reduce the mean squared error (MSE).
#Focuings on the treatment of categorical variables and the creation of
#dummy variables:

# Part 3 - Enhanced Task Code:


# Data Preparation
df_new = pd.get_dummies(df_new, columns=['state'], drop_first=True)

# Define features and target variable for the regression model
features = ['bathrooms', 'bedrooms', 'square_feet'] + [col for col in df_new if col.startswith('state_')]
target = 'price'

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_new[features], df_new[target], test_size=0.10, random_state=12345)

# Calculate the mean price for the baseline prediction
mean_price_train = y_train.mean()
baseline_predictions = [mean_price_train] * len(y_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Perform 10-fold cross-validation
rmse_scorer = make_scorer(mean_squared_error, squared=False)
cv_scores = cross_val_score(linear_model, X_train, y_train, cv=10, scoring=rmse_scorer)
average_rmse_cv = cv_scores.mean()

# Predict on the test set and calculate the RMSE
y_pred = linear_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the results
print(f"Mean Price in Training Set: {mean_price_train}")
print(f"Baseline RMSE using average price: {baseline_rmse}")
print(f"Average RMSE from cross-validation: {average_rmse_cv}")
print(f"RMSE on test set: {rmse_test}")


#I categorized the state variable, and the numbers went down, as
#766.6601706141287 went to 666.2155160460591, meaning that by grouping the state
#together reduces the predictive accuracy compared to the original set of 
#features.

#Part 4 - Decision Trees Classifier Predicting Apartment Price Ranges Using
#Key Features


#Developed a Decision Tree Classifier that predicts the price category of
#apartments. The price categories are binned based on the rental price

# Part 4 - Starter Task Code:
#Define Price Categories:
bins = [0, 1000, 2000, 3000, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'High']
df_new['price_category'] = pd.cut(df_new['price'], bins=bins, labels=labels, right=False)


#2 - Select Features and Target
X = df_new[['bedrooms', 'bathrooms', 'square_feet']]
y = df_new['price_category']


#3 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=12345)

#4 - Intialize and Train to Model
decision_tree_model = DecisionTreeClassifier(random_state=12345)

# Fit the classifier to the training data
decision_tree_model.fit(X_train, y_train)

# Predict the price categories on the test set
y_pred = decision_tree_model.predict(X_test)



#5 - output
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred)

# Output the results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", accuracy)
print("\nWeighted Precision:", precision)
print("\nWeighted Recall:", recall)
print("\nClassification Report:\n", class_report)
ConfusionMatrixDisplay(conf_matrix, display_labels=labels).plot()



#I enhanced the accuracy of my classification model.
# Part 4 - Enhanced Task Code:

#1 - Explore and Selection
bins = [0, 1000, 2000, 3000, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'High']
df_new['price_category'] = pd.cut(df_new['price'], bins=bins, labels=labels, right=False)

X = df_new[['square_feet', 'latitude', 'longitude']]
y = df_new['price_category']

#2- Feature Engineering
#3- Data prep
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=12345)

#4 Model training
decision_tree_model = DecisionTreeClassifier(random_state=12345)
decision_tree_model.fit(X_train, y_train)

# Predict and Evaluate the model
y_pred = decision_tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#the accuracy has improved with the new features

#5 - Output 
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", accuracy)
print("\nWeighted Precision:", precision)
print("\nWeighted Recall:", recall)
print("\nClassification Report:\n", class_report)

fig = px.scatter_3d(df_new, x='latitude', y='longitude', z='price_category', color='price_category')
fig.show()
ConfusionMatrixDisplay(conf_matrix, display_labels=labels).plot()


#Part 5 - Comparing Average Rental Prices to State Median Incomes
#reset data frame

df_new = pd.read_csv("~/Desktop/Itec 400/Datasets/apartments.csv")
columns_to_convert = ['price', 'square_feet', 'bathrooms', 'bedrooms', 'latitude', 'longitude']
for column in columns_to_convert:
    df_new[column] = pd.to_numeric(df_new[column], errors='coerce')

df_new.dropna(subset=columns_to_convert, inplace=True)

df_new.dropna(subset=['state'], inplace=True)

df_new = df_new[df_new['state'].str.match('^[A-Z]{2}$') == True]


df_new.reset_index(drop=True, inplace=True)


df_new


#Part 5 - Comparing Average Rental Prices to State Median Incomes
# 1- Data Aggregation:
url = "https://res.cloudinary.com/dixv5n1ye/raw/upload/v1699335440/state_income_b5yz22.json"
 
response = requests.get(url)
 

if response.status_code == 200:
   income = response.json()
   income = pd.DataFrame(income.items(), columns=['state', 'median_income'])

#rental price data grouped 
price_state = df_new.groupby('state')['price'].mean().reset_index()
 
##join
df_join = price_state.merge(income, on='state', how='inner')
df_join.info()
 
 
# Combined dataframe with rental price and income by state
df_join = price_state.merge(income, on='state', how='inner')
 
#ratio
df_join['rent_to_income_ratio'] = df_join['price'] / df_join['median_income']
 
# Sort by ratio descending
df_sorted = df_join.sort_values('rent_to_income_ratio', ascending=False)
 
print(df_sorted)
 
 
 
fig = px.bar(df_sorted, x='state', y='rent_to_income_ratio', color='state',
            title='Rent to Income Ratio by State',
            labels={'state':'State', 'rent_to_income_ratio':'Rent to Income Ratio'},
            color_discrete_sequence=px.colors.qualitative.Dark24)
 
fig.update_layout(xaxis={'categoryorder':'total descending'})            
fig.update_xaxes(tickangle=45)
 
fig.show()




