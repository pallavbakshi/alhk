import pandas as pd
import xgboost as xgb
import json

age = 68.322
species = 'Asian elephant'

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

d = pd.DataFrame(columns=cols)

d.loc[0] = colvals
d.loc[1] = colvalssecond

d = d.reindex_axis(sorted(d.columns), axis=1)

print (d)

bst = xgb.Booster()  # init model
bst.load_model('synthetic_train.model')  # load data

# Predicting the labels of the test set: preds
df_test = xgb.DMatrix(d)
preds = bst.predict(df_test)

print preds[0]
