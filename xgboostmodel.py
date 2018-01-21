import os
import pandas as pd
import xgboost as xgb
import itertools
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def predict(species, age):

    try:

        # print (os.getcwd())
        #
        # files = [ft for ft in os.listdir('.') if os.path.isfile(ft)]
        #
        # print (' '.join(files))

        with open("encoder.json", "r") as f:
            data = json.loads(f.read())

        cols = ['age']
        colvals = [age]
        colvalssecond = [1]

        print (type(species))

        print (type(age))

        for key in data:
            cols.append(str(data[key]))
            colvalssecond.append(1)
            if data[key] == species:
                colvals.append(float(1))
            else:
                colvals.append(float(0))

        d = pd.DataFrame(columns=cols)

        d.loc[0] = colvals
        d.loc[1] = colvalssecond

        d = d.reindex_axis(sorted(d.columns), axis=1)

        print (d)

        bst = xgb.Booster()  # init model
        bst.load_model('train.model')  # load data

        # Predicting the labels of the test set: preds
        df_test = xgb.DMatrix(d)
        preds = bst.predict(df_test)

        return preds[0]

    except Exception as ex:
        print (ex)



def train_model(db, Learn):
    # dataset = db.session.query(Learn).all()
    # print (dataset.column_descriptions)

    df = pd.read_sql('SELECT * FROM Learn', db.session.bind)
    df = df.drop(['id', 'create_at'], axis=1)

    # df = pd.read_csv('synthetic_data.csv')

    # print (df)
    # print (df.columns)
    # print (df.info())
    #
    # return

    try:
        # Separating features and target variable
        y = df['score']
        x = df.drop('score', axis=1)

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

        # Creating DMatrix
        df_dm = xgb.DMatrix(data=x, label=y)

        # Training the model and chossing hyperparameters & parameters

        # parameter dictionary: params
        params = {"objective": "reg:linear"}

        # Create list of hyperparameter values

        # Here I am using less params range
        # due to shortage of processing speed
        # on Heroku free version. If running locally
        # use any range that you want. Ideally, I would
        # have used the following ranges at least
        # eta_vals = [0.001, 0.01, 0.1]
        # reg_params = [1, 10]
        # max_depths = [1, 2, 5]
        # subsamples = [0.3, 0.6, 0.9]

        eta_vals = [0.1]
        reg_params = [1]
        max_depths = [2]
        subsamples = [0.9]

        list_of_params = [eta_vals, reg_params, max_depths, subsamples]
        params_vary = list(itertools.product(*list_of_params))

        # Empty array to store rmes at each iteration
        best_rmse = []

        counter = 0
        # Systematically vary params
        for curr_val in params_vary:
            # print ("counter ----------------------------------")
            # print (counter)

            counter = counter + 1

            params["eta"] = curr_val[0]
            params["lambda"] = curr_val[1]
            params["max_depth"] = curr_val[2]
            params["subsample"] = curr_val[3]

            # Perform cross-validation
            cv_results = xgb.cv(dtrain=df_dm, params=params, nfold=2, num_boost_round=20, early_stopping_rounds=5, metrics="rmse", as_pandas=True, seed=123, verbose_eval=True)

            # Append the final round rmse to best_rmse
            best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

        comb = pd.DataFrame(list(zip(params_vary, best_rmse)), columns=["params", "best_rmse"])

        # Assigning the min rmse params
        (eta, lam, max_depth, subsample) = comb.loc[comb['best_rmse'].idxmin()]['params']

        # Reformulating the params dict
        params = {"objective": "reg:linear", "eta": eta, "lambda": lam, "max_depth": max_depth, "subsample": subsample}

        # Training the model: xg_reg
        # num_boost_round not same as initial parameter upon which it was trained
        # time taken for cross-validating with 100 boosting round is too high for
        # server request. That causes request timeout. Hence the number of boosting
        # round is small in the xgb.cv function. However, they can be increased
        # if we make async call to this function, which is possible on any system
        # but here I am using free Heroku version and they don't allow me to open
        # mutiple processes
        xg_reg = xgb.train(params=params, dtrain=df_dm, num_boost_round=50)

        print ("Saving the model")
        # Save the model
        xg_reg.save_model('train.model')

        # Save the parameters
        to_write_params = ' '.join(str(x) for x in [eta, lam, max_depth, subsample]) + '\n'
        with open('params.txt', 'a') as the_file:
            the_file.write(to_write_params)

        # Save encoder classes
        encoder_list = zip(range(len(le.classes_)), le.classes_)
        data = json.dumps({key: value for (key, value) in encoder_list})
        with open("encoder.json", "w") as f:
            f.write(data)

        # print (os.getcwd())
        #
        # files = [ft for ft in os.listdir('.') if os.path.isfile(ft)]
        #
        # print (' '.join(files))
        #
        # print (os.path.dirname(os.path.realpath('params.txt')))

        print ("Training complete")

    except Exception as ex:
        print (ex)


def reset_table(db, Learn):
    try:
        dataset = db.session.query(Learn).delete()
        db.session.commit()
        return 'ok'
    except Exception as ex:
        print (ex)
        return 'failed'
