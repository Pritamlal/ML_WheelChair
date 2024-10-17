import pandas as pd
# Importing the pandas library for data manipulation and analysis

from sklearn.model_selection import train_test_split
# Importing the train_test_split function to split the dataset into training and testing sets

from sklearn.tree import DecisionTreeClassifier
# Importing the DecisionTreeClassifier class to create a decision tree model for classification tasks

from sklearn.metrics import accuracy_score, classification_report
# Importing functions to evaluate the performance of the model. `accuracy_score` gives the accuracy, 
# and `classification_report` provides detailed metrics like precision, recall, and F1-score

from sklearn.preprocessing import LabelEncoder
# Importing the LabelEncoder class to convert categorical labels into numerical values 
# (useful when the target or features contain non-numeric values)

import joblib
# Importing the joblib library to save and load the trained machine learning models (model persistence)
# Load hand landmarks data for the first gesture

gesture1_data = pd.read_csv('idle.csv')
gesture1_data['Gesture'] = 0  # Assign a label (0) for the first gesture

# Load hand landmarks data for the second gesture
gesture2_data = pd.read_csv('forward.csv')
gesture2_data['Gesture'] = 1  # Assign a label (1) for the second gesture

# Load hand landmarks data for the third gesture
gesture3_data = pd.read_csv('backward.csv')
gesture3_data['Gesture'] = 2  # Assign a label (2) for the third gesture

# Load hand landmarks data for the third gesture
gesture4_data = pd.read_csv('left.csv')
gesture4_data['Gesture'] = 3  # Assign a label (3) for the third gesture

# Load hand landmarks data for the third gesture
gesture5_data = pd.read_csv('right.csv')
gesture5_data['Gesture'] = 4  # Assign a label (4) for the third gesture

# Concatenate data from all gestures
combined_data = pd.concat([gesture1_data, gesture2_data, gesture3_data, gesture4_data, gesture5_data], ignore_index=True)

# Assuming the last column contains the gesture labels
target_column = 'Gesture'

if target_column in combined_data.columns:
    print(f"Found target column: {target_column}")

    #convert string labels to numerical values
    label_encoder = LabelEncoder()
    combined_data[target_column] = label_encoder.fit_transform(combined_data[target_column])

    # Use the identified target column
    X = combined_data.iloc[:, :-1]  # Features (all columns except the last one)
    y = combined_data[target_column]  # Labels (last column)

    print(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose a model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'gesture_recognition_model.joblib')

    # Predict
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print(classification_report(y_test, y_pred))

    # you can use model.predict(new_data) where new_data is a DataFrame with hand landmarks features
else:
    print("Target column not found. Please check your data.")