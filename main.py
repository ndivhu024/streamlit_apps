import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import io

import streamlit as st

st.set_page_config(layout="wide")

st.title("Employee Attrition Prediction with Decision Tree Classification")

st.header("Project Overview")
st.write(
"This project applies a supervised machine learning Decision Tree Classification algorithm, to build a model that predicts whether "
"employees are at risk of being terminated or leaving their jobs."
)

st.header("Problem Statement")
st.write(
"Organizations invest significant resources in recruiting and training employees. "
"High employee turnover can therefore lead to substantial financial and productivity "
"losses. By identifying factors that influence employee retention, companies can "
"take proactive steps to reduce turnover and improve workforce stability."
)

st.header("Dataset")
st.write(
    "This project uses a synthetic Human Resources dataset "
    "[(Source: Huebner & Patalano, 2019)](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set). "
    "This dataset is used to train a model that predicts whether an employee is at risk of termination or voluntary resignation. "
    "If early indicators of employee attrition can be identified, organizations may be able to intervene and retain valuable talent."
)

#open the whole data set
df = pd.read_csv('HRDataset.csv', delimiter = ",")
#st.dataframe(df)


column = st.multiselect('Filter Table', df.columns, df.columns[0:6])

filter_df = df[column]
st.dataframe(filter_df,hide_index=True)


st.header('Variables used to train model')

st.text('Termd it is a categorical response variable and the rest are predictors')

data = {
    "Variable": [
        "MarriedID",
        "DeptID",
        "PerfScoreID",
        "FromDiversityJobFairID",
        "PayRate",
        "Termd",
        "EmpSatisfaction",
        "SpecialProjectsCount"
    ],
    
    "Description": [
        "Is the employee married? (1 for yes, 0 for no)",
        "The department ID code from 1 to 6 that matches the department the employee works in",
        "The performance score code that matches the employee’s most recent performance score",
        "Was the employee sourced from the diversity job fair? (1 for yes, 0 for no)",
        "The employee’s hourly pay rate (all salaries are converted to hourly pay rate)",
        "Has this employee been terminated? (1 for yes, 0 for no)",
        "A basic satisfaction score between 1 and 5 from a recent employee satisfaction survey",
        "The number of special projects the employee worked on during the past 6 months"
    ]
}

df_variables = pd.DataFrame(data)

st.dataframe(df_variables,  hide_index=True)


#st.text('Identify the type of data in each column')
#buffer = io.StringIO()
#df.info(buf=buffer)
#st.text(buffer.getvalue())

#st.text('Explore reasons for termination')
#st.write(df['EmploymentStatus'].unique())

st.text('Check for null values')

all_vars = ["MarriedID", "PerfScoreID", "FromDiversityJobFairID", "PayRate", "EmpSatisfaction",
 "SpecialProjectsCount", "Termd"]

# Check that there are no missing values
df_to_use1 = df.loc[:, all_vars]
st.write(df_to_use1.isnull().sum())

with st.expander("One hot encoding"):
    st.write(" DeptID is an unordered categorical predictor, meaning there is no ordinal relation among its classes (1–6). To prevent the decision tree algorithm from treating it as an ordered variable, I apply One-Hot Encoding. This transforms DeptID into separate binary columns for each class, where 1 indicates membership in a particular class and 0 indicates absence. See added DeptID_0 to DeptID_5 columns in the dataset table below")
ohe = OneHotEncoder(categories='auto')
Xd = ohe.fit_transform(df.DeptID.values.reshape(-1, 1)).toarray()
df_ohe = pd.DataFrame(Xd, columns = ["DeptID_"+str(int(i)) for i in range(Xd.shape[1])])

# Add encoded feature to the dataframe
df_to_use2 = pd.concat([df_to_use1, df_ohe], axis=1)
st.dataframe(df_to_use2.head())

#Split Dataset into Features (X) and Target Variable (y)"
X = df_to_use2.iloc[:, np.r_[0:6, 7:13]]
y = df_to_use2.loc[:, ["Termd"]]



st.header('Training the model')

st.text("Split the dataset into training and testing subsets. These are complementary to each other; for example, setting 25% of the data as the test set leaves 75% for training. You can adjust the test set size using the slider below.")

test_size = st.slider('Set test subset size', 0.1, 0.5, 0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size ,random_state = 0)

st.subheader('Fitting training dataset to a tree-based classification model')
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

st.code("""
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
""", language="python")

st.write(f"Maximum Tree depth ={classifier.get_depth()} and Number of leaves = {classifier.get_n_leaves()} for the fitted model.")
      
st.text('The depth of a tree is the longest path from the root node to a leaf node. It measures how many splits the tree makes before reaching a prediction. A small depth → simple tree → may underfit (miss patterns). A large depth → complex tree → may overfit (fit noise in training data). The number of leaves is the total number of terminal nodes in the tree. More leaves → more detailed predictions → risk of overfitting.Fewer leaves → simpler predictions → risk of underfitting')      
      
max_depth = st.slider(
    "Set the depth of the decision tree to display. \n"
    "Note: Increasing the depth makes the tree more complex and the diagram larger, "
    "which makes labels harder to read.",
    1,
    classifier.get_depth(),
    1
)

# Create the figure
fig, ax = plt.subplots(figsize=(12,6), dpi=300)
#plt.tight_layout()

# Plot the decision tree
#set class in order of sorted(unique(y))
plot_tree(classifier,max_depth=max_depth,filled=True, feature_names=list(X.columns), class_names=['not terminated', 'terminated'], ax=ax)

# Display in Streamlit
st.pyplot(fig, use_container_width=True)

y_pred = classifier.predict(X_test)
test_score = accuracy_score(y_test, y_pred)

st.subheader('Accuracy Score')

st.text('Accuracy score measures the accuracy of a model to make correct predictions of the response variable.')

st.code(
    '''
y_pred = classifier.predict(X_test)
test_score = accuracy_score(y_test, y_pred)
''',
    language='python'
)

st.write("Accuracy score of the tree = {:2.2%}".format(test_score),'this is not that bad for an uncontrained model')


#st.header('Plot the full tree')
#plt.figure() 
#plot_tree(classifier,feature_names=list(X.columns))
#plt.show()


st.header('Constraining model')

st.text("The current model accuracy is moderate, but still relatively low given the tree depth and number of leaves obtained from the training data. To improve its performance on unseen data (generalization) and avoid overfitting where the model simply memorises patterns in the training set the tree must be constrained. In this project, I constrain the model using a minimum number of samples per leaf node. This ensures that each split affects a sufficient number of observations, reducing the influence of outliers and preventing the model from memorizing noise in the training data."
)

st.subheader('The accuracy of the model on the training and test sets for varying minimum leaf sample sizes')

samples = [sample for sample in range(1,30)]     
classifiers = []
for sample in samples:
    classifier2 = DecisionTreeClassifier(random_state=0, 
                                         min_samples_leaf=sample)
    classifier2.fit(X_train, y_train)
    classifiers.append(classifier2)
    
train_scores = [clf.score(X_train, y_train) for clf in classifiers]
test_scores = [clf.score(X_test, y_test) for clf in classifiers]

fig, ax = plt.subplots(figsize = (6,3))
ax.set_xlabel("Minimum leaf samples")
ax.set_ylabel("Accuracy")
ax.set_title("Comparing the training and test set accuracy")
ax.plot(samples, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(samples, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
st.pyplot(fig, use_container_width=True)

with st.expander('code'):
    st.code("""
    # Sample sizes for minimum leaf nodes
    samples = [sample for sample in range(1, 30)]

    # Train a Decision Tree with varying min_samples_leaf
    classifiers = []
    for sample in samples:
        classifier2 = DecisionTreeClassifier(random_state=0, min_samples_leaf=sample)
        classifier2.fit(X_train, y_train)
        classifiers.append(classifier2)

    # Calculate training and test accuracy
    train_scores = [clf.score(X_train, y_train) for clf in classifiers]
    test_scores = [clf.score(X_test, y_test) for clf in classifiers]

    # Plot accuracy vs. minimum leaf samples
    fig, ax = plt.subplots()
    ax.plot(samples, train_scores, marker='o', label="Train", drawstyle="steps-post")
    ax.plot(samples, test_scores, marker='o', label="Test", drawstyle="steps-post")
    ax.set_xlabel("Minimum leaf samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Test Accuracy for Different Min Leaf Samples")
    ax.legend()
    st.pyplot(fig)
    """, language="python")

st.subheader('Cross Validation')

st.text("In order to find the optimal minimum leaf samples cross validation is applied. Cross validation is a technique to evaluate how well a model generalises to new data. It works by splitting the dataset into multiple parts (called folds). The model is trained on some folds and tested on the remaining fold. This process is repeated so that each fold is used as a test set once. The results from all folds are averaged to give a more reliable estimate of model performance, minimum leaf sample with the highest validation score in our case.")

st.subheader('The cross-validated minimum leaf samples')


with st.expander("Code"):
    st.code("""
validation_scores = []

for sample in samples:
    classifier3 = DecisionTreeClassifier(random_state=1, min_samples_leaf=sample)
    score = cross_val_score(estimator=classifier3, X=X_train, y=y_train, cv=cv)
    validation_scores.append(score.mean()
    
    fig, ax = plt.subplots(figsize=(10,3))

    ax.set_xlabel("Minimum leaf samples")
    ax.set_ylabel("Validation score")
    ax.set_title("Validation scores at different minimum leaf sample counts")

    # Plot the validation scores
    ax.plot(samples, validation_scores, marker='o', label="Validation", drawstyle="steps-post")

    ax.legend()
    st.pyplot(fig)
    
)
""", language="python")

cv = st.slider('Cross-validation folds', 1, 10, 5)

validation_scores = []
for sample in samples:
    classifier3 = DecisionTreeClassifier(random_state=1, min_samples_leaf=sample)
    score = cross_val_score(estimator=classifier3, X=X_train, y=y_train, cv=cv)   
    validation_scores.append(score.mean())
    

fig, ax = plt.subplots(figsize=(6,3))

ax.set_xlabel("Minimum leaf samples")
ax.set_ylabel("Validation score")
ax.set_title("Validation scores at different minimum leaf sample counts")

# Plot the validation scores
ax.plot(samples, validation_scores, marker='o', label="Validation", drawstyle="steps-post")

ax.legend()
st.pyplot(fig)


samples_optimum = samples[validation_scores.index(max(validation_scores))]

st.text(f"The minimum leaf samples per node with the highest cross validation =  {samples_optimum}")

st.subheader('The constrained model')

st.text('Fitting the cross validated optimum minimum sample size to the whole dataset (X,y)')

best_model = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum) #samples_optimum = 5
best_model.fit(X, y)
y_predicted = best_model.predict(X)
test_score_final = accuracy_score(y_predicted, y)
st.write(
    "Accuracy score of the optimal tree = {:2.2%}. This is a huge improvement from the accuracy score of the unconstrained tree = {:2.2%}."
    .format(test_score_final, test_score)
)
st.write("Tree depth =",best_model.get_depth(),'\n'
      "Number of leaves =",best_model.get_n_leaves(), '\n' 'and Minimum sample size per node=', samples_optimum)

max_depth = st.slider('Set maximum tree depth to display', 1, best_model.get_depth(),2)

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Plot the decision tree
plot_tree(
    best_model,
    max_depth=max_depth,
    feature_names=list(X.columns),
    class_names=['not terminated', 'terminated'],
    impurity=True,
    filled=True,
    ax=ax         
)

st.pyplot(fig,use_container_width=True)

with st.expander("Code"):
    st.code("""
best_model = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum)  # samples_optimum = 5
best_model.fit(X, y)

y_predicted = best_model.predict(X)
test_score_final = accuracy_score(y_predicted, y)

st.write("Accuracy score of the optimal tree = {:2.2%}".format(test_score_final))
st.write("Tree depth =", best_model.get_depth(), '\\n'
         "Number of leaves =", best_model.get_n_leaves())

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

# Plot the decision tree
plot_tree(
    best_model,
    max_depth=2,
    feature_names=list(X.columns),
    class_names=['not terminated', 'terminated'],
    impurity=False,
    filled=True,
    ax=ax
)

st.pyplot(fig)
""", language="python")

st.subheader("Next: Explore Cost Complexity Pruning")

st.write("""
**Cost Complexity Pruning (CCP)** is a technique that helps simplify decision trees 
to prevent overfitting. A very deep tree can fit the training data perfectly but 
perform poorly on unseen dataset. CCP introduces a **complexity parameter (alpha)** that 
trades off **tree size** versus **accuracy**:

- A **larger alpha** value prunes more branches, resulting in a **smaller, simpler tree**.
- A **smaller alpha** keeps more branches, allowing a **more complex tree**.

By testing different alpha values, we can find a tree that balances **accuracy** and 
**interpretability**, often improving predictions on unseen data.
""")

