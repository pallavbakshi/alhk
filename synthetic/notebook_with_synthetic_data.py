import pandas as pd
import xgboost as xgb
import numpy as np
import itertools
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('synthetic_data.csv')

# Separating features and target variable
x, y = df.iloc[:, :-1], df.iloc[:, -1]

df = x

# LabelEncoder object: le
le = LabelEncoder()

df['proxy_species'] = pd.DataFrame(le.fit_transform(df['species']))

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(sparse=False)

# df_encoded
df_encoded = pd.DataFrame(ohe.fit_transform(df['proxy_species'].values.reshape(-1, 1)))

# Renaming columns
df_encoded.columns = le.classes_

# Joining df_encoded to df
df = df.join(df_encoded)

# Dropping proxy_species and species
df = df.drop(['species', 'proxy_species'], axis=1)

df = df.reindex_axis(sorted(df.columns), axis=1)

x = df

test = False
load = False
singlepoint = True

if test:
    # Uncomment for train test split ---------
    # Splitting into train and test ~ Hide
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=111)



# Comment below if using train test split ----------

if not test:
    # Creating DMatrix
    df_dm = xgb.DMatrix(data=x, label=y)

if test:
    # Uncomment for train test split ---------
    df_dm = xgb.DMatrix(data=x_train, label=y_train)
    df_dm_test = xgb.DMatrix(data=x_test)

if not load:
    # Comment below if loading the model ---------

    # Training the model and chossing hyperparameters & parameters

    # parameter dictionary: params
    params = {"objective": "reg:linear"}

    # Create list of hyperparameter values
    eta_vals = [0.1]
    reg_params = [1]
    max_depths = [2]
    subsamples = [0.9]

    # eta_vals = [0.001, 0.01, 0.1]
    # reg_params = [1, 10]
    # max_depths = [1, 2, 5]
    # subsamples = [0.3, 0.6, 0.9]

    list_of_params = [eta_vals, reg_params, max_depths, subsamples]
    params_vary = list(itertools.product(*list_of_params))

    print len(params_vary)

    # Empty array to store rmes at each iteration
    best_rmse = []

    # Systematically vary params
    for curr_val in params_vary:

        params["eta"] = curr_val[0]
        params["lambda"] = curr_val[1]
        params["max_depth"] = curr_val[2]
        params["subsample"] = curr_val[3]

        # Perform cross-validation
        cv_results = xgb.cv(dtrain=df_dm, params=params, nfold=10, num_boost_round=100, early_stopping_rounds=25, metrics="rmse", as_pandas=True, seed=123, verbose_eval=None)

        # Append the final round rmse to best_rmse
        best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

    comb = pd.DataFrame(list(zip(params_vary, best_rmse)),columns=["params","best_rmse"])

    # Assigning the min rmse params
    (eta, lam, max_depth, subsample) = comb.loc[comb['best_rmse'].idxmin()]['params']

    # print
    print (comb.loc[comb['best_rmse'].idxmin()]['params'])
    print (comb.loc[comb['best_rmse'].idxmin()])

    # Reformulating the params dict
    params = {"objective":"reg:linear", "eta": eta, "lambda": lam, "max_depth": max_depth, "subsample": subsample}

    # Training the model: xg_reg
    xg_reg = xgb.train(params=params, dtrain=df_dm, num_boost_round=100)

    # Save the model
    xg_reg.save_model('synthetic_train.model')

    # Save the parameters
    to_write_params = ' '.join(str(x) for x in [eta, lam, max_depth, subsample]) + '\n'
    with open('synthetic_params.txt', 'a') as the_file:
        the_file.write(to_write_params)

    print (type(le.classes_[0]))

    print ("df columns")
    print (df.columns)

    encoder_list = zip(range(len(le.classes_)), le.classes_)
    data = json.dumps({key: value for (key, value) in encoder_list})
    with open("synthetic_encoder.json", "w") as f:
        f.write(data)


if load and test:
    # Uncomment below to load the model and predict with testing dataset --------
    # load model
    bst = xgb.Booster()  # init model
    bst.load_model('synthetic_trained.model')  # load data

    # Predicting the labels of the test set: preds
    preds = bst.predict(df_dm_test)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

if singlepoint and not load:
    age = 41.6
    species = 'Chimpanzee'

    with open("synthetic_encoder.json", "r") as f:
        data = json.loads(f.read())

    cols = ['age']
    colvals = [age]
    colvalssecond = [1]

    print (type(species))

    print (type(age))

    for key in data:
        cols.append(str(data[key]))
        print type(data[key])
        colvalssecond.append(1)
        if data[key] == species:
            colvals.append(float(1))
        else:
            colvals.append(float(0))

    print (cols)
    print (colvals)

    d = pd.DataFrame(columns=cols)

    d.loc[0] = colvals
    d.loc[1] = colvalssecond

    d = d.reindex_axis(sorted(d.columns), axis=1)

    print (d)

    print (d.columns)

    # Predicting the labels of the test set: preds
    df_test = xgb.DMatrix(d)
    preds = xg_reg.predict(df_test)

    print preds[0]
    print preds[1]
