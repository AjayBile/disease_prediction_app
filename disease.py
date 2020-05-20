import pandas as pd
import pickle
import warnings
import os, sys
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')

# for tree binarisation
from sklearn.tree import DecisionTreeClassifier

# to build the models
from sklearn.ensemble import AdaBoostClassifier

# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data():
    data = pd.read_csv("cardio_train.csv", sep=';')
    return data


def convert_year(cols):
    age_in_days = cols[0]
    return int(age_in_days / 365)


def get_outliers(df, colname):
    IQR = df[colname].quantile(0.75) - df[colname].quantile(0.25)

    upper_fence = df[colname].quantile(0.75) + (IQR * 3)
    lower_fence = df[colname].quantile(0.25) - (IQR * 3)

    return upper_fence, lower_fence


def data_processing():
    df = load_data()

    print(df.columns.tolist())

    # convert age into age_year
    df['age_yr'] = df[['age']].apply(convert_year, axis=1)

    # drop previous age column
    df.drop(['id', 'age'], inplace=True, axis=1)

    temp = df.copy()
    temp.drop(["cardio"], axis=1, inplace=True)

    # find the numerical and categorical columns

    numerical_cols = []
    category_cols = []

    for col in temp.columns.tolist():
        if len(temp[col].unique()) > 15:
            numerical_cols.append(col)
        else:
            category_cols.append(col)

    for numcol in numerical_cols:
        upper_boundary, lower_boundary = get_outliers(temp, numcol)

        temp.loc[temp[numcol] > upper_boundary, numcol] = upper_boundary
        temp.loc[temp[numcol] < lower_boundary, numcol] = lower_boundary

    dataset = temp.copy()

    # convert gender 2 i.e. female value as 0
    dataset['gender'].replace({2: 0}, inplace=True)

    # apply one hot encoder on categorical columns
    dataset = pd.get_dummies(temp, columns=category_cols)

    dataset['cardio'] = df['cardio']

    return dataset


def train():
    dataset = data_processing()

    feature_list = dataset.columns.tolist()
    feature_list.remove('cardio')

    X = dataset[feature_list]
    Y = dataset[['cardio']]

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

    ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=0.5)
    ada_model.fit(X_train, Y_train.values.ravel())

    # Save Model As Pickle File
    with open('ada_model.pkl', 'wb') as m:
        pickle.dump(ada_model, m)

    test(X_test, Y_test)


# Test accuracy of the model
def test(X_test, Y_test):
    with open('ada_model.pkl', 'rb') as mod:
        p = pickle.load(mod)

    print(X_test.columns.tolist())
    pre = p.predict(X_test)
    # print(roc_auc_score(Y_test, pre))   # Prints the roc-auc of the model
    print(accuracy_score(Y_test, pre))  # Prints the accuracy of the model


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def processed_incoming_data(userdict):
    dict_for_prediction = \
        {
            "height": 0, "weight": 0, "ap_hi": 0, "ap_lo": 0, "age_yr": 0, "gender_1": 0, "gender_2": 0,
            "cholesterol_1": 0, "cholesterol_2": 0, "cholesterol_3": 0, "gluc_1": 0, "gluc_2": 0, "gluc_3": 0,
            "smoke_0": 0, "smoke_1": 0, "alco_0": 0, "alco_1": 0, "active_0": 0, "active_1": 0
        }

    for feature in userdict.keys():
        if feature in ["age_yr", "height", "weight", "ap_hi", "ap_lo"]:
            dict_for_prediction[feature] = userdict[feature]

    for feature in userdict.keys():
        if feature in ["cholesterol", "gluc", "smoke", "alco", "active", "gender"]:
            if userdict[feature] == 0:
                dict_for_prediction[feature + "_0"] = 1
            elif userdict[feature] == 1:
                dict_for_prediction[feature + "_1"] = 1
            elif userdict[feature] == 2:
                dict_for_prediction[feature + "_2"] = 1
            elif userdict[feature] == 3:
                dict_for_prediction[feature + "_3"] = 1

    print(dict_for_prediction)
    return dict_for_prediction


def predict_on_user_input(data: dict):
    row_for_prediction: dict = processed_incoming_data(data)
    df = pd.DataFrame(data=row_for_prediction, index=[0])
    print(df.head())
    with open(find_data_file('ada_model.pkl'), 'rb') as model:
        p = pickle.load(model)
    op = p.predict(df)
    return op[0]


# first time you should run with train method so your model file will generate.
# print(predict_on_user_input(
#     {"age_yr": 54, "gender": 1, "height": 167, "weight": 70, "ap_hi": 140, "ap_lo": 90, "cholesterol": 2, "gluc": 3,
#      "smoke": 0, "alco": 1, "active": 1}))
