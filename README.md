## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Open the jupyter notebook.
2. Create the new->python library
3. write the python code in the notebook
4. Execute and run the program
 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

file_path = 'Employee.csv'
data = pd.read_csv(file_path)

data = pd.get_dummies(data)

X = data.drop('left', axis=1)
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()

```

## Output:

<img width="1347" height="774" alt="Screenshot 2026-05-13 091242" src="https://github.com/user-attachments/assets/96162bbe-2e08-4493-9223-38e9c89d2f79" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
