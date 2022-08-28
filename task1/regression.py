################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

import ast
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from joblib import dump, load
import math
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_K = 7
DEFAULT_N_ESTIMATORS = 51
DEFAULT_MIN_SAMPLES_SPLIT = 15
DEFAULT_DEPTH = 3
TRAIN_P = 0.75
top_values = []


def convert_to_list(lst):
    lst = list(ast.literal_eval(lst))
    return [d["name"] for d in lst]


def create_dummies_from_json(data_frame, feature):
    data_frame[feature] = data_frame[feature].fillna("[]")
    data_frame[feature] = data_frame[feature].apply(convert_to_list)
    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    dummies = pd.DataFrame(mlb.fit_transform(data_frame[feature]), columns=mlb.classes_)
    dummies = dummies.add_prefix(feature + "_")
    data_frame = data_frame.join(dummies)
    data_frame = data_frame.drop(feature, axis=1)
    return data_frame


def create_dummies_with_top(data_frame, feature, num):
    data_frame[feature] = data_frame[feature].fillna("[]")
    data_frame[feature] = data_frame[feature].apply(convert_to_list)
    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    dummies = pd.DataFrame(mlb.fit_transform(data_frame[feature]), columns=mlb.classes_)
    dummies = dummies.add_prefix(feature + "_")
    sums = list(dummies.sum(axis=0))
    top = sorted(range(len(sums)), key=lambda sub: sums[sub])[-num:]
    dummies = dummies.iloc[:, top]
    data_frame = data_frame.join(dummies)
    return data_frame.drop(feature, axis=1)


def find_top(data_frame, feature, num):
    dummies = data_frame[feature]
    dummies = pd.get_dummies(dummies, columns=[feature])
    sums = dummies.sum(axis=0)
    num = min(num, sums.shape[0])
    sums = list(dummies.sum(axis=0))
    top = sorted(range(len(sums)), key=lambda sub: sums[sub])[-num:]
    top_values = list(dummies.iloc[:, top].columns)
    return data_frame, top_values


def find_top_values(data_frame, feature, num):
    data_frame[feature] = data_frame[feature].fillna("[]")
    data_frame[feature] = data_frame[feature].apply(convert_to_list)
    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    dummies = pd.DataFrame(mlb.fit_transform(data_frame[feature]), columns=mlb.classes_)
    sums = dummies.sum(axis=0)
    num = min(num, sums.shape[0])
    sums = list(dummies.sum(axis=0))
    top = sorted(range(len(sums)), key=lambda sub: sums[sub])[-num:]
    top_values = list(dummies.iloc[:, top].columns)
    return data_frame, top_values


def count_top(lst):
    counter = 0
    for i in lst:
        if i in top_values:
            counter += 1
    return counter


def preprocess_test(csv_file, response_type):
    # deal with all drop features
    data_frame = pd.read_csv(csv_file)
    data_frame = data_frame.drop("id", axis=1)
    data_frame = data_frame.drop("original_title", axis=1)
    data_frame = data_frame.drop("overview", axis=1)
    data_frame = data_frame.drop("tagline", axis=1)
    data_frame = data_frame.drop("keywords", axis=1)
    data_frame = data_frame.drop("title", axis=1)

    # deals with runtime
    df_for_calculations = data_frame[(data_frame.runtime > 10)]
    median = df_for_calculations["runtime"].median()
    data_frame["runtime"] = data_frame["runtime"].apply(lambda x: median if x == 0 else x)

    # deals with languages
    data_frame["spoken_languages"] = data_frame["spoken_languages"].apply(convert_to_list)
    data_frame["spoken_languages"] = data_frame["spoken_languages"].apply(lambda x: len(x))

    data_frame["belongs_to_collection"] = data_frame["belongs_to_collection"].apply(
        lambda x: 0 if x == np.nan else 1)

    # deals with date
    data_frame.insert(data_frame.shape[1], "release_month",
                      data_frame["release_date"].apply(lambda x: pd.to_datetime(x).month))
    data_frame["release_date"] = data_frame["release_date"].apply(lambda x:
                                                                  (pd.to_datetime("today") - pd.to_datetime(x,
                                                                                                            errors='coerce')).days)
    data_frame = data_frame.rename(columns={'release_date': 'days_since_release'})

    # filters according to status of movie
    data_frame = data_frame.drop("status", axis=1)

    data_frame, top_values = find_top_values(data_frame, "production_companies", 3)
    data_frame["production_companies"] = data_frame["production_companies"].apply(count_top)

    data_frame, top_values = find_top_values(data_frame, "crew", 100)
    data_frame["crew"] = data_frame["crew"].apply(count_top)

    data_frame, top_values = find_top_values(data_frame, "cast", 100)
    data_frame["cast"] = data_frame["cast"].apply(count_top)

    data_frame, top_values = find_top(data_frame, "original_language", 5)
    data_frame["original_language"] = data_frame["original_language"].apply(count_top)

    data_frame["production_countries"] = data_frame["production_countries"].apply(convert_to_list)
    data_frame["production_countries"] = data_frame["production_countries"].apply(lambda x: len(x))

    # deals with genres
    data_frame["genres"] = data_frame["genres"].apply(convert_to_list)
    data_frame["genres"] = data_frame["genres"].apply(lambda x: len(x))

    y = None

    if response_type == "revenue":
        # deals with budget
        df_for_calculations = data_frame[(data_frame.budget > 0)]
        median = df_for_calculations["budget"].median()
        data_frame["budget"] = data_frame["budget"].apply(lambda x: median if x == 0 else x)
        data_frame["homepage"] = data_frame["homepage"].apply(lambda x: 0 if x == np.nan else 1)

        data_frame = data_frame.drop("vote_count", axis=1)

        data_frame = data_frame.fillna(0)
        # y = data_frame["revenue"]
        # y = None

    elif response_type == "vote_average":
        # deals with vote_count
        df_for_calculations = data_frame[(data_frame.vote_count > 0)]
        median = df_for_calculations["vote_count"].median()
        data_frame["vote_count"] = data_frame["vote_count"].apply(lambda x: median if x == 0 else x)

        data_frame = data_frame.drop("budget", axis=1)
        data_frame = data_frame.drop("homepage", axis=1)
        data_frame = data_frame.drop("days_since_release", axis=1)

        data_frame = data_frame.fillna(0)
        # y = data_frame["vote_average"]
        # y = None

    # data_frame = data_frame.drop("vote_average", axis=1)
    # return data_frame.drop("revenue", axis=1), y
    return data_frame


def preprocess_train(csv_file, response_type):
    global top_values

    # deal with all drop features
    data_frame = pd.read_csv(csv_file)
    data_frame = data_frame.drop("id", axis=1)
    data_frame = data_frame.drop("original_title", axis=1)
    data_frame = data_frame.drop("overview", axis=1)
    data_frame = data_frame.drop("tagline", axis=1)
    data_frame = data_frame.drop("keywords", axis=1)
    data_frame = data_frame.drop("title", axis=1)

    # deals with runtime
    df_for_calculations = data_frame[(data_frame.runtime > 10)]
    median = df_for_calculations["runtime"].median()
    data_frame["runtime"] = data_frame["runtime"].apply(lambda x: median if x == 0 else x)

    # deals with languages
    data_frame["spoken_languages"] = data_frame["spoken_languages"].apply(convert_to_list)
    data_frame["spoken_languages"] = data_frame["spoken_languages"].apply(lambda x: len(x))
    data_frame = data_frame[(data_frame.spoken_languages > 0)]

    data_frame["belongs_to_collection"] = data_frame["belongs_to_collection"].apply(
        lambda x: 0 if x == np.nan else 1)

    # deals with date
    data_frame.insert(data_frame.shape[1], "release_month",
                      data_frame["release_date"].apply(lambda x: pd.to_datetime(x).month))
    data_frame["release_date"] = data_frame["release_date"].apply(lambda x:
                                                                  (pd.to_datetime("today") - pd.to_datetime(x,
                                                                                                            errors='coerce')).days)
    data_frame = data_frame.rename(columns={'release_date': 'days_since_release'})

    # filters according to status of movie
    data_frame = data_frame[data_frame.status == "Released"]
    data_frame = data_frame.drop("status", axis=1)

    data_frame, top_values = find_top_values(data_frame, "production_companies", 3)
    data_frame["production_companies"] = data_frame["production_companies"].apply(count_top)

    data_frame, top_values = find_top_values(data_frame, "crew", 100)
    data_frame["crew"] = data_frame["crew"].apply(count_top)

    data_frame, top_values = find_top_values(data_frame, "cast", 100)
    data_frame["cast"] = data_frame["cast"].apply(count_top)

    data_frame, top_values = find_top(data_frame, "original_language", 5)
    data_frame["original_language"] = data_frame["original_language"].apply(count_top)

    data_frame["production_countries"] = data_frame["production_countries"].apply(convert_to_list)
    data_frame["production_countries"] = data_frame["production_countries"].apply(lambda x: len(x))
    data_frame = data_frame[(data_frame.production_countries > 0)]

    # deals with genres
    data_frame["genres"] = data_frame["genres"].apply(convert_to_list)
    data_frame["genres"] = data_frame["genres"].apply(lambda x: len(x))

    y = None
    if response_type == "revenue":
        # deals with budget
        df_for_calculations = data_frame[(data_frame.budget > 0)]
        median = df_for_calculations["budget"].median()
        data_frame["budget"] = data_frame["budget"].apply(lambda x: median if x == 0 else x)
        data_frame["homepage"] = data_frame["homepage"].apply(lambda x: 0 if x == np.nan else 1)
        data_frame = data_frame.drop("vote_count", axis=1)
        data_frame = data_frame.dropna()
        y = data_frame["revenue"]

    if response_type == "vote_average":
        # deals with vote_count
        df_for_calculations = data_frame[(data_frame.vote_count > 0)]
        median = df_for_calculations["vote_count"].median()
        data_frame["vote_count"] = data_frame["vote_count"].apply(lambda x: median if x == 0 else x)

        data_frame = data_frame.drop("budget", axis=1)
        data_frame = data_frame.drop("homepage", axis=1)
        data_frame = data_frame.drop("days_since_release", axis=1)

        data_frame = data_frame.dropna()
        y = data_frame["vote_average"]

    data_frame = data_frame.drop("vote_average", axis=1)
    return data_frame.drop("revenue", axis=1), y


def k_fold(X, y, models, k=DEFAULT_K):
    """
    returns estimated error for given model using k-fold cross validation
    :param X: design matrix
    :param y: response vector
    :param models: array of models to be valuated
    :param k: num of folds
    :return: best model
    """
    model_scores = dict.fromkeys(models, 0)  # model scores dictionary
    for i, model in enumerate(model_scores):
        print("model", i)
        cv = KFold(n_splits=k, random_state=1, shuffle=True)
        scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        model_scores[model] = np.mean(scores)
    return model_scores


def mse(p, r):
    return np.square(p - r).mean(0)


# plots mse on validation set, from that choose best model
def plot_mse(models, response_type):
    X, y = preprocess_test("new_train.csv", response_type)
    X, y = X.to_numpy(), y.to_numpy()
    train_X, train_y = X[:int(X.shape[0] * 0.8)], y[:int(X.shape[0] * 0.8)]
    test_X, test_y = X[int(X.shape[0] * 0.8):], y[int(X.shape[0] * 0.8):]

    # fit models, plot MSE
    curr_min_mse = math.inf
    best_model = 0
    for model in models:
        MSE = []
        for p in range(1, 101):
            i = int(len(train_X) * (p / 100))
            model.fit(train_X[:i], train_y[:i])
            MSE.append(mse(model.predict(test_X), test_y))
        # plot MSE vs. p%
        plt.plot(np.arange(1, 101), MSE)
        plt.show()
        if MSE[-1] < curr_min_mse:
            best_model = model
            curr_min_mse = MSE[-1]
    return best_model


def baseline_model():
    pass
    # model = S


def choose_and_train_model(response_type):
    # get data
    X, y = preprocess_train("new_train.csv", response_type)

    # play with hyper-parameters to get models
    models = [AdaBoostRegressor(), GradientBoostingRegressor()]

    for alpha in [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10]:
        models += [Ridge(alpha=alpha)]
        models += [Lasso(alpha=alpha)]

    for depth in range(1, 10):
        for splits in range(4, 17):
            models += [RandomForestRegressor(n_estimators=DEFAULT_N_ESTIMATORS, max_depth=depth,
                                             min_samples_split=splits, bootstrap=True)]

    # add resnet?
    # receives a matrix, outputs a vector - known sizes. use some non-linear shit!

    # run k-fold to choose best one, then save it
    k_fold_sorted = k_fold(X, y, models)
    best_models = dict((sorted(k_fold_sorted.items(), key=lambda item: item[1]))[:5])
    print(best_models)
    best_model = plot_mse(best_models, response_type)
    best_model.fit(X, y)
    if response_type == 'revenue':
        dump(best_model, 'revenue_model.joblib')
    else:
        dump(best_model, 'vote_average_model.joblib')


def do_everything():
    choose_and_train_model("revenue")
    choose_and_train_model("vote_average")


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    # load pickle -> trained model for revenue and vote
    revenue_model = load('revenue_model.joblib')
    vote_average_model = load('vote_average_model.joblib')

    # preprocess the data with the same preprocess function
    x1 = preprocess_test(csv_file, "revenue")
    x2 = preprocess_test(csv_file, "vote_average")

    yhat1 = revenue_model.predict(x1)
    yhat2 = vote_average_model.predict(x2)

    # assert edge cases per vote and revenue: 0 workers, time in the future and so on

    data_frame = pd.read_csv(csv_file)
    # print(yhat1.shape)
    # print(data_frame.shape[0])
    yhat1[data_frame["status"] != "Released"] = 0
    yhat2[data_frame["status"] != "Released"] = 0

    yhat1[x1["production_countries"] == 0] = 0
    yhat2[x2["production_countries"] == 0] = 0

    # change to return only yhats
    return yhat1.tolist(), yhat2.tolist()


def test():
    yhat1, yhat2 = predict("test.csv")
    print(yhat1, yhat2)
    # print(mse(yhat1, y1), mse(yhat2, y2))


# do_everything()
# test()
