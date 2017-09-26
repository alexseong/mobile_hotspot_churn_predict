import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def clean_train(df):
    # Drop 'signup_date'
    df = df.drop('signup_date', axis=1)

    # Fill null values in 'avg_rating_by_driver' with mean of non-null values
    by_driver_fill = np.mean(df['avg_rating_by_driver'])
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(by_driver_fill)

    # Fill null values in 'avg_rating_of_driver' with 0s
    of_driver_fill = np.mean(df['avg_rating_of_driver'])
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(of_driver_fill)

    # Fill null values in 'phone' with mode of non-null values
    phone_fill = scs.mode(df['phone'])[0][0]
    df['phone'] = df['phone'].fillna(phone_fill)

    # Dummify 'phone' column
    phone_dummies = pd.get_dummies(df['phone'])
    df[phone_dummies.columns] = phone_dummies
    df = df.drop('phone', axis=1)

    # Dummify 'city' column
    city_dummies = pd.get_dummies(df['city'])
    df[city_dummies.columns] = city_dummies
    df = df.drop('city', axis=1)

    return df, by_driver_fill, of_driver_fill, phone_fill, phone_dummies.columns.tolist(), city_dummies.columns.tolist()


def clean_test(df, by_driver_fill, of_driver_fill, phone_fill, phone_columns, city_columns):
    # Drop 'signup_date'
    df = df.drop('signup_date', axis=1)

    # Fill null values in 'avg_rating_by_driver' with mean of non-null values
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(by_driver_fill)

    # Fill null values in 'avg_rating_of_driver' with 0s
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(of_driver_fill)

    # Fill null values in 'phone' with mode of non-null values
    df['phone'] = df['phone'].fillna(phone_fill)

    # Dummify 'phone' column
    phone_dummies = pd.get_dummies(df['phone'])
    df[phone_dummies.columns] = phone_dummies
    df = df.drop('phone', axis=1)

    # Dummify 'city' column
    city_dummies = pd.get_dummies(df['city'])
    df[city_dummies.columns] = city_dummies
    df = df.drop('city', axis=1)

    all_cols = set(df.columns)
    for col in phone_columns + city_columns:
        if col not in all_cols:
            df[col] = 0

    return df


if __name__ == '__main__':
    df = pd.read_csv('data/churn.csv')

    df['Churn'] = (pd.to_datetime(df['last_trip_date']) < '2014-06-01').astype(int)
    df.drop('last_trip_date', axis=1, inplace=True)

    y = df['Churn']
    X = df.ix[:, df.columns != 'Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    X_train, by_driver_fill, of_driver_fill, phone_fill, phone_columns, city_columns = clean_train(X_train)

    keep = ['iPhone', 'Android']

    # Logistic Regression
    models = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), KNeighborsClassifier()]

    for model in models:
        score = cross_val_score(model, X_train, y_train, scoring='recall', cv=5)
        print '{}: {}'.format(model.__class__.__name__, score.mean())

    X_test = clean_test(X_test, by_driver_fill, of_driver_fill, phone_fill, phone_columns, city_columns)
