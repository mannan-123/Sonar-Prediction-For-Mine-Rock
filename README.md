# Sonar Rock vs Mine Classification using Logistic Regression

## Overview

This project implements a **Logistic Regression** model to classify sonar signals as either **Rock (R)** or **Mine (M)**. The dataset used is the **Sonar Data** set, which consists of sonar signals reflected off metal cylinders (mines) and rocks.

## Installation

To run this project, you need Python and the following libraries installed:

```bash
pip install numpy pandas scikit-learn
```

## Importing Necessary Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Collection and Preprocessing

```python
dataset = pd.read_csv('/content/Sonar data.csv', header=None)

# Display first 5 rows
dataset.head()

# Shape of the dataset
dataset.shape

# Statistical summary
dataset.describe()

# Count unique values in the first column
dataset[0].value_counts()

# Splitting features and target variable
Y = dataset[60]
X = dataset.drop(columns=60, axis=1)
```

## Training and Testing Data

```python
# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Initializing and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Model Evaluation

```python
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data: ', test_data_accuracy)
```

## Making Predictions

```python
# Sample input data for inference
model_input = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]

# Convert input to numpy array
model_input = np.asarray(model_input).reshape(1, -1)

# Make prediction
prediction = model.predict(model_input)

# Output result
if prediction[0] == "R":
    print("It is a rock")
else:
    print("It is a mine")
```

## Results

- The model is trained and evaluated on the **Sonar Data** dataset.
- The accuracy scores on the training and testing data are 83% and 76% respectively.
- The model predicts whether a given sonar signal corresponds to a **rock** or a **mine**.

## License

This project is open-source and free to use.
