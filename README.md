# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Create a small employee dataset with Level, Position_CLevel, and Salary columns.
2. Split the data into features (X = Level and Position_CLevel) and target (y = Salary).
3. Train a DecisionTreeClassifier on the entire dataset (X, y) so it can learn the exact salary classes.
4. Predict the salaries using the same training data to get the y_pred values.
5. Evaluate the model by printing the accuracy score and classification report.
6. Visualize the trained decision tree using plot_tree() with the feature names.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AMRITHA VARSHINI M 
RegisterNumber: 212224250002
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

df = pd.DataFrame({
    "Level": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "Position_CLevel": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Salary": [45000, 50000, 60000, 80000, 110000, 150000, 200000, 
               300000, 500000, 1000000, 2000000, 3000000, 4000000]
})

X = df.drop("Salary", axis=1)
y = df["Salary"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred) * 100)
print("\nClassification Report:\n", classification_report(y, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Regressor")
plt.show()

```

## Output:
<img width="573" height="488" alt="image" src="https://github.com/user-attachments/assets/75806ab9-9999-463e-b976-d25c4faf3e23" />
<img width="1570" height="812" alt="image" src="https://github.com/user-attachments/assets/cf11f2c3-1e11-455f-be56-003a325407e5" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
