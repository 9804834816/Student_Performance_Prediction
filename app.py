import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    confusion_matrix
)

# st.set_page_config() set properties of webpage
# This changes title shown on browser tab
# add icon on brower tab 
st.set_page_config(
    page_title = "Student Performance Predictor🎓",
    page_icon ="",
    layout = "wide"
)


#st.markdown() is used to display Html,Css
# .stApp means entire Streamlit application
# h1 is first heading ,p is paragraph text , label of input boxes
# data-testid ="stSidebar" is predefined attribute used by streamlit to apply css to sidebar 
st.markdown("""
<style>
.stApp{
background-color : black;
}
h1,h2,h3,h4,h5,h6,p,label
{
    color:white ! important;
}
[data-testid="stSidebar"] 
{
background-color : #2E2E2E;
}
</style>
""",
unsafe_allow_html=True
)

#load dataset
df = pd.read_csv("file (2).csv")
total_students = len(df)
pass_student = (df["Pass_Fail"] == "Pass").sum()
fail_student = (df["Pass_Fail"] == "Fail").sum() 
# create sidebar
st.sidebar.title("Model Selection")
algorithm = st.sidebar.selectbox(
    "Choose An Algorithm",
    ["Logistic Regression",
     "Decision Tree",
     "Random Forest",
     "Support Vector Machine",
     "K-Nearest Neighbors",
     "Naive Bayes"
     ]
)
page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Dataset Preview",
        "Dataset Summary",
        "Data Analysis",
        "Model Training",
        "Prediction"
    ]
)
st.sidebar.image(
    "books.jpg"
)
# If user selected home page then content of it will be displayed
if page == "Home":
   st.title("Student Performance Prediction Model")
   st.write("This Model Predict Whether student Pass/Fail")
   col1 , col2 , col3 ,col4 = st.columns(4)
# first Metric card
   with col1 :
       st.metric("Total Students",total_students)
# Second Metric card
   with col2 :
       st.metric("Pass Students",pass_student)
#Third Metric card
   with col3 :
       st.metric("Fail Students",fail_student)
#Fourth Metric card
   with col4 :
       st.metric("Total Features ",df.shape[1])
   st.header("About Project")
   st.write(""" Student Performance Model is used to predict whether student will Pass/Fail 
based on different academic and personal factor""")
#dataset Overview
   st.subheader("Dataset Overview") # display medium size heading
   st.dataframe(df.head(3)) # Give user quick preview of dataset by displaying first 3 rows
   st.write("Dataset Shape:",df.shape) # return size of dataset(rows,columns)

if page == "Dataset Preview":
    st.title("Dataset Preview")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Column Names")
        st.dataframe(df.columns,height=300) # display names of all columns in table
    with col2:
        st.subheader("Project Workflow")
        st.image("image.jpg",width =800)
    st.subheader("Dataset Information")
    st.write("Dataset Shape :", df.shape)
    st.write("Missing Values :", df.isnull().sum().sum())
    st.write("Target Column :", "Pass_Fail")

if page == "Dataset Summary":
    st.title("Dataset Summary")
    st.write("This Session provides overall Statistical summary of dataset . It help to analyze distribution of numerical features .")
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

if page == "Data Analysis":
    st.title("Data Analysis")
    pivot = pd.pivot_table(
    df,
    values="Obtained_Marks",
    index="Pass_Fail",
    aggfunc="mean"
    )
    st.write(pivot)
    fig,axis = plt.subplots(figsize=(5,3))
    pivot.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black",
    linewidth=4,
    ax = axis
    )
    axis.set_title("Average Obtained Marks", fontweight="bold")
    axis.set_xlabel("Pass/Fail", fontweight="bold")
    axis.set_ylabel("Average Marks", fontweight="bold")
    st.pyplot(fig)

# model training
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
y = y.map({"Pass":1,"Fail":0})
if page == "Model Training":

    st.title("Model Training")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Logistic Regression
    if algorithm == "Logistic Regression":

        lr_model = LogisticRegression(max_iter=1000)

        lr_model.fit(x_train, y_train)

        prediction = lr_model.predict(x_test)

        logistic_accuracy = accuracy_score(y_test, prediction)

        st.subheader("Accuracy")
        st.write(logistic_accuracy)

        st.subheader("Confusion Matrix")

        fig, axis = plt.subplots(figsize=(5,3))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            cmap="Greens",
            ax=axis
        )

        st.pyplot(fig)

        st.subheader("Classification Report")

        st.text(classification_report(
            y_test,
            prediction
        ))

    # Decision Tree
    elif algorithm == "Decision Tree":
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(x_train, y_train)
        prediction = dt_model.predict(x_test)
        tree_accuracy = accuracy_score(y_test, prediction)
        st.subheader("Accuracy")
        st.write(tree_accuracy)
        st.subheader("Decision Tree")
        fig, axis = plt.subplots(figsize=(5,3))
        tree.plot_tree(
            dt_model,
            feature_names=x.columns,
            class_names=["Fail","Pass"],
            filled=True,
            ax=axis
        )
        st.pyplot(fig)
        st.subheader("Confusion Matrix")
        fig, axis = plt.subplots(figsize=(5,3))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            cmap="Blues",
            ax=axis
        )
        st.pyplot(fig)

    # Random Forest
    elif algorithm == "Random Forest":
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(x_train, y_train)
        prediction = rf_model.predict(x_test)
        random_forest_accuracy = accuracy_score(
            y_test,
            prediction
        )

        st.subheader("Accuracy")
        st.write(random_forest_accuracy)
        st.subheader("Feature Importance")
        importance = rf_model.feature_importances_
        fig, axis = plt.subplots(figsize=(5,3))
        axis.barh(
            x.columns,
            importance,
            color="yellow",
            edgecolor="black",
            linewidth=2
        )
        axis.set_title("Feature Importance")
        axis.set_xlabel("Importance Score")
        axis.set_ylabel("Features")
        st.pyplot(fig)

    # Support Vector Machine
    elif algorithm == "Support Vector Machine":
        svm_model = SVC(kernel="linear")
        svm_model.fit(x_train, y_train)
        prediction = svm_model.predict(x_test)
        svm_accuracy = accuracy_score(
            y_test,
            prediction
        )
        st.subheader("Accuracy")
        st.write(svm_accuracy)
        st.subheader("Student Performance Graph")
        fig, axis = plt.subplots(figsize=(5,3))
        axis.scatter(
            df["Internet_Hours"],
            df["Study_Hours"],
            c=df["Pass_Fail"].map({"Pass":"blue","Fail":"red"})
        )
        axis.set_title("Student Performance")
        axis.set_xlabel("Internet Hours")
        axis.set_ylabel("Study Hours")
        st.pyplot(fig)
    # KNN
    elif algorithm == "K-Nearest Neighbors":
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        prediction = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(
            y_test,
            prediction
        )
        st.subheader("Accuracy")
        st.write(knn_accuracy)
        st.subheader("Confusion Matrix")
        fig, axis = plt.subplots(figsize=(5,3))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            cmap="Oranges",
            ax=axis
        )
        st.pyplot(fig)
    # Naive Bayes
    elif algorithm == "Naive Bayes":
        nb_model = GaussianNB()
        nb_model.fit(x_train, y_train)
        prediction = nb_model.predict(x_test)
        nb_accuracy = accuracy_score(
            y_test,
            prediction
        )
        st.subheader("Accuracy")
        st.write(nb_accuracy)
        st.subheader("Confusion Matrix")
        fig, axis = plt.subplots(figsize=(5,3))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            cmap="Purples",
            ax=axis
        )
        st.pyplot(fig)

# prediction page
# Train Logistic Regression Model
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, # 80% training ,20% testing
    random_state=42
)
# train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)
if page == "Prediction":
    st.title("Student Performance")
    # Create two equal columns
    col1, col2 = st.columns([1, 1])# both columns get 50% of page width
    # Left Column
    with col1:
        st.subheader("Enter Student Details")
        Study_Hours = st.slider("Study Hours", 0, 12)
        Attendance = st.slider("Attendance", 0, 100)
        Obtained_Marks = st.slider("Obtained Marks", 0, 600)
        Previous_Year_Marks = st.slider("Previous Year Marks", 0, 600)
        Sleep_Hours = st.slider("Sleep Hours", 0, 12)
        Internet_Hours = st.slider("Internet Hours", 0, 12)
        Extra_Activities = st.slider("Extra Activities", 0, 10)
        predict = st.button("Predict")
    # Right Column 
    with col2:
        st.subheader("Prediction Result")
        if predict:
            user_data = [[
                Study_Hours,
                Attendance,
                Obtained_Marks,
                Previous_Year_Marks,
                Sleep_Hours,
                Internet_Hours,
                Extra_Activities
            ]]
            result = lr_model.predict(user_data)
            if result[0] == 1: # result[0]means first number
                st.success("Student Will Pass")# if predicted value is 1 
            else:
                st.error("Student Will Fail")# if predicted value is 0 
        # Pie Chart
        pass_count = (df["Pass_Fail"] == "Pass").sum()
        fail_count = (df["Pass_Fail"] == "Fail").sum()
        st.subheader("Pass vs Fail")
        fig, axis = plt.subplots(figsize=(4, 4))# axis is area inside figure where pie chart is drawn
        axis.pie(
            [pass_count, fail_count],
            labels=["Pass", "Fail"],
            autopct="%1.1f%%" # display percent with one decimal place
        )
        axis.set_title("Student Performance")
        st.pyplot(fig)