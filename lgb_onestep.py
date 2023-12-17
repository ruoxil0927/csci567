from utils import *
from dataset import *
from train import train_gbm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Load dataset
df_train, df_test, df_sales, df_promo, df_sales_store, df_promo_store, df_sales_item, df_promo_item = load_promo_and_sales('./dataset')
df_store = load_supplementary('./dataset')
df_store = df_store.reindex(df_promo.index.get_level_values(0))

# Build train and test data
num_days = 21
t_train = pd.date_range(pd.Timestamp('2017-7-19') - pd.DateOffset(days=(num_days - 1) * 7),
                        pd.Timestamp('2017-7-19'),
                        freq='7D')
t_val = pd.Timestamp('2017-7-26')
t_test = pd.Timestamp('2017-8-16')

# training data
X_l, y_l = [], []
for t in t_train:
    X_store_item = prepare_dataset(df_sales, df_promo, t)
    y_i = df_sales.loc[:, t: t + pd.DateOffset(days=15)].values

    # X_item = prepare_dataset(df_sales_item, df_promo_item, t, name_prefix='item')
    # X_item.index = df_sales_item.index
    # X_item = X_item.reindex(df_sales.index.get_level_values(1)).reset_index(drop=True)

    X_i = pd.concat([X_store_item, df_store.reset_index()], axis=1)

    X_l.append(X_i)
    y_l.append(y_i)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

# validation data
X_val_store_item = prepare_dataset(df_sales, df_promo, t_val)

# X_item_val = prepare_dataset(df_sales_item, df_promo_item, t_val, name_prefix='item')
# X_item_val.index = df_sales_item.index
# X_item_val = X_item_val.reindex(df_sales.index.get_level_values(1)).reset_index(drop=True)

X_val = pd.concat([X_val_store_item, df_store.reset_index()], axis=1)
y_val = df_sales.loc[:, t_val: t_val + pd.DateOffset(days=15)].values

# testing data
X_test_store_item = prepare_dataset(df_sales, df_promo, t_test)

# X_item_test = prepare_dataset(df_sales_item, df_promo_item, t_test, name_prefix='item')
# X_item_test.index = df_sales_item.index
# X_item_test = X_item_test.reindex(df_sales.index.get_level_values(1)).reset_index(drop=True)

X_test = pd.concat([X_test_store_item, df_store.reset_index()], axis=1)

# Define model params
gbm_params = {
    'model': {
        'num_leaves': 31,
        'objective': 'regression',
        'min_data_in_leaf': 10,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'metric': 'rmse',
        'num_threads': 16
    },
    'boost_round': 2000,
    'early_stopping_rounds': 100
}

# Train model
print("Training...")
models, val_pred, test_pred = train_gbm(X_train, y_train, X_val, y_val, X_test, gbm_params)

y_pred = np.array(val_pred).transpose()
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation rmse: {val_rmse}")

# Make prediction file
print("Make submission...")
make_submission(df_train, df_test, df_sales, test_pred, zero_forecast=True, name='gbm_submission.csv')

err = (np.array(val_pred).transpose() - y_val) ** 2
df_err = pd.DataFrame(
    err, index=df_sales.index,
    columns=pd.date_range("2017-07-26", periods=16).to_period('D')
).stack().to_frame("sales")
df_err.index.set_names(['store_nbr', 'family', 'date'], inplace=True)
df_err = df_err.unstack(level=['store_nbr', 'date'])
df_err = np.sqrt(df_err.sum(axis=1) / df_err.shape[1])
print(df_err)



"""
err = (np.array(val_pred).transpose() - y_val) ** 2
err = err.sum(axis=1)
top_40 = np.argsort(err)[::-1][:40]
stores_num = top_40 // 33
items_num = top_40 % 33

y_pred = np.array(val_pred).transpose()
df_val = pd.DataFrame(
    y_pred, index=df_sales.index,
    columns=pd.date_range("2017-07-26", periods=16).to_period('D')
).stack().to_frame("sales")
df_val.index.set_names(['store_nbr', 'family', 'date'], inplace=True)


for store, family in zip(stores_num, items_num):
    store = 1 if store == 0 else store
    plot_pred(df_promo, df_sales, df_val, store=store, family=family)

err = (np.array(val_pred).transpose() - y_val) ** 2
df_err = pd.DataFrame(
    err, index=df_sales.index,
    columns=pd.date_range("2017-07-26", periods=16).to_period('D')
).stack().to_frame("sales")
df_err.index.set_names(['store_nbr', 'family', 'date'], inplace=True)
df_err = df_err.unstack(level=['store_nbr', 'date'])
df_err = df_err.sum(axis=1) / df_err.shape[1]
print(df_err)


print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_sales.index,
    columns=pd.date_range("2017-08-16", periods=16).to_period('D')
).stack().to_frame("sales")
df_preds.index.set_names(['store_nbr', 'family', 'date'], inplace=True)


# merge
pred1 = pd.read_csv('./submission.csv')
submission = df_test[["id"]].join(df_preds, how="left").fillna(0.)
submission['sales'] = np.clip(np.expm1(submission['sales']), 0, None)

submission = submission.merge(pred1, how='left', on='id')
submission.index = df_test.index
for i in range(1, 55):
    submission.loc[(i, 31), 'sales_x'] = submission.loc[(i, 31), 'sales_y'].values


submission = submission.sort_values(by='id')
submission.to_csv('lgb_sub.csv', float_format='%.6f', index=None)
"""