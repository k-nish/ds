# coding:utf-8
import numpy as np
import pandas as pd
try:
    import MySQLdb as mysql
except ImportError:
    import mysql.connector as mysql

def learning(clf,X,Y):
    clf.fit(X,Y)
    return clf

def predict(clf,X,Y,evaluate):
    predict = clf.predict(X)
    prediction_score = evaluate(Y,predict)
    print (prediction_score)
    return prediction_score

def standardscaler(X):
    from sklearn.preprocessing import StandardScaler
    model = StandardScaler()
    X = model.fit_transform(X)
    return X

def minmaxscaler(X):
    from sklearn.preprocessing import MinMaxScaler
    model = MinMaxScaler()
    X = model.fit_transform(X)
    return X

def corrcoef_all_row(X,Y):
    array = np.array([])
    for i in range(0,X.shape[1]):
        a = X[:,i]
        b = Y
        var = np.corrcoef(a,b)[0][1]
        array = np.append(array,var)
    return array

def classifier(train_X,train_Y,valid_X,valid_Y,evaluate):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    import xgboost as xgb

    clfs = [LinearSVC(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            LogisticRegression(),
            xgb.XGBClassifier(),
            GaussianNB(),
            KNeighborsClassifier()]
    results = np.array([])
    for clf in clfs:
        print '-'**50
        print clf
        clf.fit(train_X,train_Y)
        print ('score_train:' + evaluate(train_Y, clf.predict(train_X)))
        print ('score_valid:' + evaluate(valid_Y, clf.predict(valid_X)))
        results = np.append(results,evaluate(valid_Y, clf.predict(valid_X)))
    max_index = np.argmax(results)

    # validatioのスコアが最もよい学習器を返す
    best_clf = clfs[max_index]
    return best_clf, results[max_index]

def regression(train_X,train_Y,valid_X,valid_Y,evaluate):
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    from pyfm import pylibfm

    clfs = [SVR(), RandomForestRegressor(), GradientBoostingRegressor(), LogisticRegression(), xgb.XGBRegressor(),pylibfm.FM(task="regression")]
    results = np.array([])
    for clf in clfs:
        print '-'**50
        print (str(clf))
        clf.fit(train_X,train_Y)
        print ('score_train:' + evaluate(train_Y, clf.predict(train_X)))
        print ('score_valid:' + evaluate(valid_Y, clf.predict(valid_X)))
        results = np.append(results,evaluate(valid_Y, clf.predict(valid_X)))

    max_index = np.argmax(results)
    # validatioのスコアが最もよい学習器を返す
    best_clf = clfs[max_index]
    return best_clf

def prediction(train_X,train_Y,valid_X,valid_Y,test_X,test_Y,clf, evaluation):
    clf.fit(train_X,train_Y)
    # train
    pred_train = clf.predict(train_X)
    score_train = evaluation(train_Y,pred_train)
    # valid
    pred_valid = clf.predict(valid_X)
    score_valid = evaluation(valid_Y,pred_valid)
    # test
    pred_test = clf.predict(test_X)
    score_test = evaluation(test_Y,pred_test)
    print ('train:' + str(score_train))
    print ('valid:' + str(score_valid))
    print ('test:' + str(score_test))

    return clf


def gridsearch_classify(algorithm, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, grid, scoring, evaluation):
    try:
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        from sklearn.grid_search import GridSearchCV


    # グリッドサーチとモデル学習
    def classify(algorithm, train_X, train_Y, grid, scoring):
        clf = GridSearchCV(algorithm, grid, scoring)
        clf.fit(train_X, train_Y)
        model = clf.best_estimator_
        return [model, clf.best_estimator_]

    def valuation(model, X, Y):
        prediction = model.predict(X)
        score = evaluation(Y, prediction)
        print (score)
        return score

    # 分類モデルの学習
    model, param = classify(algorithm, train_X, train_Y, grid, scoring)
    print (param)
    # 過学習チャック
    train_score = valuation(model, train_X, train_Y)
    valid_score = valuation(model, valid_X, valid_Y)
    test_score = valuation(model, test_X, test_Y)

    return [model, train_score, valid_score, test_score]
