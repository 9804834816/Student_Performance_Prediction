import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load Dataset
df = pd.read_csv("file (2).csv")
print(df.head())   # by default display first five rows
print(df.info())   # shows number of rows and columns, datatype

# Pivot Table
# values="Obtained_Marks" tells pandas to calculate only obtained marks column
# index="Pass_Fail" groups data into two categories: Pass and Fail
# aggfunc="mean" gives average marks of failed and passed students
pivot = pd.pivot_table(
    df,
    values="Obtained_Marks",
    index="Pass_Fail",
    aggfunc="mean"
)
print(pivot)
pivot.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black",
    linewidth=4
)
plt.title("Average Obtained Marks", fontweight="bold")
plt.xlabel("Pass/Fail", fontweight="bold")
plt.ylabel("Average Marks", fontweight="bold")
plt.show()

print(df.describe())

# X stores input features
# Select only those columns which help ML model to predict
# Leave Name and Roll_Number because they do not help in training
x = df[
[
"Study_Hours",
"Attendance",
"Obtained_Marks",
"Previous_Year_Marks",
"Sleep_Hours",
"Internet_Hours",
"Extra_Activities"
]
]
# y stores target variable
y = df["Pass_Fail"]
print("Input Feature of x:")
print(x.head())
print("Target Feature of y:")
print(y.head())

# Convert Target Variable into Numbers
# Machine Learning algorithms cannot work directly with text
# Pass = 1
# Fail = 0
y = y.map({"Pass":1,"Fail":0})
print("Converted Text Into Numerical")
print(y.head())

# Split Dataset into Training and Testing
# 0.2 means 20% testing data and 80% training data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print(x_train.shape)
print(x_test.shape)

# Logistic Regression is supervised ML Classification Algorithm
# used to predict two possible outcomes (binary classification)
# It is used when target/output variable has only two classes
lr_model = LogisticRegression(max_iter=1000)
# max_iter means maximum number of times logistic regression is allowed to learn from training data
lr_model.fit(x_train, y_train)
# fit() is used to train model
# Model learns relationship between input features and target
prediction = lr_model.predict(x_test)
# predict() uses testing data for prediction
print(prediction)

logistic_accuracy=accuracy_score(y_test,prediction)# it will compare actual answers with prediction answers and tell how accurate our machine model
print("Accuracy:",logistic_accuracy)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    prediction,
    cmap="Greens"
)
plt.show()
print(classification_report(
    y_test,
    prediction
))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
# create a decision tree classification model
# x_train => training input feature
# y_train => training target feature
# Model learns pattern from training data
dt_model.fit(x_train, y_train)
# dt_model => draw trained decision tree
# feature_names=x.columns => takes all column names from feature DataFrame x
# filled=True => fill all boxes with colors
tree.plot_tree(
    dt_model,
    feature_names=x.columns,
    class_names=["Fail", "Pass"],
    filled=True
)
plt.show()

# Gini formula:
# 1 - (P(Fail)^2 + P(Pass)^2)
# predict() output from testing input features
prediction = dt_model.predict(x_test)
print(prediction)
tree_accuracy = accuracy_score(
    y_test,
    prediction
)
print("Decision Tree Accuracy:", tree_accuracy)
ConfusionMatrix = confusion_matrix(
    y_test,
    prediction
)
print(ConfusionMatrix)


# Random Forest
# ensemble is a module
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(
    x_train,
    y_train
)
prediction = rf_model.predict(x_test)
print(prediction)
random_forest_accuracy = accuracy_score(
    y_test,
    prediction
)
print("Random Forest Accuracy",random_forest_accuracy)
importance = rf_model.feature_importances_
print(importance)
feature_names = x.columns
# horizontal bar
plt.barh(
    feature_names,
    importance,
    color="yellow",
    edgecolor="black",
    linewidth=6
)
plt.title("Feature Importance(Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# SVC means Support Vector Classifier
# It is used for classification problems like Pass/Fail
# kernel tells SVM how to separate data
svm_model = SVC(kernel="linear")
# linear word in lower case
svm_model.fit(
    x_train,
    y_train
)
prediction = svm_model.predict(x_test)
print(prediction)

SVM_accuracy = accuracy_score(
    y_test,
    prediction
)
print("Support Vector Machine Accuracy:", SVM_accuracy)
plt.scatter(
    df["Internet_Hours"],
    df["Study_Hours"],
    c=df["Pass_Fail"].map({"Pass":"blue","Fail":"Red"})
)
plt.title("Student Performance Based on Internet_Hours & Study_Hours")
plt.xlabel("Internet Hours")
plt.ylabel("Study Hours")
plt.show()


# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
# n_neighbors=5 means model looks at 5 nearest students
# before predicting whether student will pass/fail
knn_model.fit(
    x_train,
    y_train
)
prediction = knn_model.predict(x_test)
print(prediction)
knn_accuracy = accuracy_score(
    y_test,
    prediction
)
print("Accuracy of KNN:", knn_accuracy)
ConfusionMatrix = confusion_matrix(
    y_test,
    prediction
)
print(ConfusionMatrix)


# GaussianNB is Naive Bayes classifier used when input features are numerical
nb_model = GaussianNB()
# create a model
# model learns relationship between input features and target variable
nb_model.fit(
    x_train,
    y_train
)
prediction = nb_model.predict(x_test)
print(prediction)
nb_accuracy = accuracy_score(
    y_test,
    prediction
)
print("Accuracy of Naive Bayes:", nb_accuracy)
ConfusionMatrix = confusion_matrix(
    y_test,
    prediction
)
print(ConfusionMatrix)

# Accuracy Comparison Graph
algorithms = [
    "Logistic_Reg.",
    "Decision",
    "Random Forest",
    "SVM",
    "KNN",
    "Naive Bayes"
]
accuracy = [
    logistic_accuracy,
    tree_accuracy,
    random_forest_accuracy,
    SVM_accuracy,
    knn_accuracy,
    nb_accuracy
]
plt.bar(
    algorithms,
    accuracy,
    color="red",
    edgecolor="black",
    linewidth=7
)
plt.title(
    "Accuracy Comparison among all Algorithms",
    fontweight="bold"
)
plt.xlabel(
    "Algorithms"
)
plt.ylabel(
    "Accuracy"
)
plt.show()