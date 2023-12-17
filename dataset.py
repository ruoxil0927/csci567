import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


def load_promo_and_sales(path):
    df_train = pd.read_csv(
        os.path.join(path, 'train.csv'),
        converters={'sales': lambda x: np.log1p(float(x))},
        parse_dates=["date"],
        infer_datetime_format=True
    )
    df_train['date'] = df_train.date.dt.to_period('D')

    df_test = pd.read_csv(
        os.path.join(path, 'test.csv'),
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    df_test['date'] = df_test.date.dt.to_period('D')

    # Encode categorical data
    df_train['family'] = le.fit_transform(df_train['family'].values)
    df_test['family'] = le.fit_transform(df_test['family'].values)

    df_train = df_train[df_train.date >= '2015']
    df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()
    df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

    # Combine promo info in train and test dataset
    promo_train = df_train[['onpromotion']].unstack(level=-1)
    promo_train.columns = promo_train.columns.get_level_values(1)

    promo_test = df_test[['onpromotion']].unstack(level=-1)
    promo_test.columns = promo_test.columns.get_level_values(1)
    df_promo = pd.concat([promo_train, promo_test], axis=1)

    # Sales info
    df_sales = df_train[['sales']].unstack(level=-1)
    df_sales.columns = df_sales.columns.get_level_values(1)

    df_sales_store = df_sales.groupby('store_nbr')[df_sales.columns].mean()
    df_promo_store = df_promo.groupby('store_nbr')[df_promo.columns].mean()

    df_sales_item = df_sales.groupby('family')[df_sales.columns].mean()
    df_promo_item = df_promo.groupby('family')[df_promo.columns].mean()

    return df_train, df_test, df_sales, df_promo, df_sales_store, df_promo_store, df_sales_item, df_promo_item


def load_supplementary(path):
    # Load supplementary data
    stores = pd.read_csv(
        os.path.join(path, 'stores.csv'),
    ).set_index('store_nbr')

    stores['city'] = le.fit_transform(stores['city'].values)
    stores['state'] = le.fit_transform(stores['state'].values)
    stores['type'] = le.fit_transform(stores['type'].values)

    return stores


def prepare_dataset(df_sales, df_promo, t, name_prefix=None):
    weight_factor = 0.8
    X = {}

    # Promo and sales feature in last i days window
    for i in [3, 7, 14, 30, 60]:
        last_idays_sales = df_sales.loc[:, t - pd.DateOffset(days=i): t - pd.DateOffset(days=1)]
        last_idays_promo = df_promo.loc[:, t - pd.DateOffset(days=i): t - pd.DateOffset(days=1)]

        last_idays_promo_sales = last_idays_sales[last_idays_promo.astype(bool)].fillna(0.)
        last_idays_no_promo_sales = last_idays_sales[~last_idays_promo.astype(bool)].fillna(0.)

        last_week_idays_sales = df_sales.loc[:, t - pd.DateOffset(days=i + 7): t - pd.DateOffset(days=8)]

        X[f'last_{i}days_promo_sales_mean'] = last_idays_promo_sales.mean(axis=1).values
        X[f'last_{i}days_promo_sales_decay'] = (last_idays_promo_sales * np.power(weight_factor, np.arange(last_idays_promo_sales.shape[1])[::-1])).sum(axis=1).values
        X[f'last_{i}days_promo_sales_rate_mean'] = (last_idays_promo_sales / last_idays_promo).fillna(0.).mean(axis=1).values

        X[f'last_{i}days_no_promo_sales_mean'] = last_idays_no_promo_sales.mean(axis=1).values
        X[f'last_{i}days_no_promo_sales_decay'] = (last_idays_no_promo_sales * np.power(weight_factor, np.arange(last_idays_no_promo_sales.shape[1])[::-1])).sum(axis=1).values

        X[f'last_{i}days_sales_diff_mean'] = last_idays_sales.diff(axis=1).mean(axis=1).values
        X[f'last_{i}days_sales_decay'] = (last_idays_sales * np.power(weight_factor, np.arange(last_idays_sales.shape[1])[::-1])).sum(axis=1).values
        X[f'last_{i}days_sales_mean'] = last_idays_sales.mean(axis=1).values
        X[f'last_{i}days_sales_median'] = last_idays_sales.median(axis=1).values
        X[f'last_{i}days_sales_min'] = last_idays_sales.min(axis=1).values
        X[f'last_{i}days_sales_max'] = last_idays_sales.max(axis=1).values
        X[f'last_{i}days_sales_std'] = last_idays_sales.std(axis=1).values

        X[f'last_week_{i}days_sales_diff_mean'] = last_week_idays_sales.diff(axis=1).mean(axis=1).values
        X[f'last_week_{i}days_sales_decay'] = (last_week_idays_sales * np.power(weight_factor, np.arange(last_week_idays_sales.shape[1])[::-1])).sum(axis=1).values
        X[f'last_week_{i}days_sales_mean'] = last_week_idays_sales.mean(axis=1).values
        X[f'last_week_{i}days_sales_median'] = last_week_idays_sales.median(axis=1).values
        X[f'last_week_{i}days_sales_min'] = last_week_idays_sales.min(axis=1).values
        X[f'last_week_{i}days_sales_max'] = last_week_idays_sales.max(axis=1).values
        X[f'last_week_{i}days_sales_std'] = last_week_idays_sales.std(axis=1).values

        X[f'total_sale_days_in_last_{i}days'] = (last_idays_sales > 0).sum(axis=1).values
        X[f'last_sales_day_in_last_{i}days'] = i - ((last_idays_sales > 0) * np.arange(last_idays_sales.shape[1])).max(axis=1).values
        X[f'first_sales_day_in_last_{i}days'] = ((last_idays_sales > 0) * np.arange(last_idays_sales.shape[1], 0, -1)).max(axis=1).values

        X[f'total_promo_days_in_last_{i}days'] = (last_idays_promo > 0).sum(axis=1).values
        X[f'last_promo_day_in_last_{i}days'] = i - ((last_idays_promo > 0) * np.arange(last_idays_promo.shape[1])).max(axis=1).values
        X[f'first_promo_day_in_last_{i}days'] = ((last_idays_promo > 0) * np.arange(last_idays_promo.shape[1], 0, -1)).max(axis=1).values

    for i in range(1, 16):
        X[f'sales_{i}days_ago'] = df_sales.loc[:, t - pd.DateOffset(days=i)].values.ravel()

    for i in range(7):
        same_day_in_last_4week = pd.date_range(t - pd.DateOffset(days=28 - i), periods=4, freq='7D')
        same_day_in_last_4week = same_day_in_last_4week.strftime('%Y-%m-%d')
        X[f'mean_4_dow{i}_2017'] = df_sales.loc[:, same_day_in_last_4week].mean(axis=1).values

    # Promo and sales feature in future i days window
    for i in [3, 7, 14]:
        futuer_idays_promo = df_promo.loc[:, t + pd.DateOffset(days=1): t + pd.DateOffset(days=i)]
        X[f'future_{i}days_promo'] = futuer_idays_promo.sum(axis=1).values

    future_15days_promo = df_promo.loc[:, t + pd.DateOffset(days=1): t + pd.DateOffset(days=15)]
    X['future_15days_promo_days'] = (future_15days_promo > 0).sum(axis=1).values
    X['future_15days_last_promo_day'] = 15 - ((future_15days_promo > 0) * np.arange(15)).max(axis=1).values
    X['future_15days_first_promo_day'] = ((future_15days_promo > 0) * np.arange(15, 0, -1)).max(axis=1).values

    for i in range(-16, 16):
        X[f'promo_{i}'] = df_promo.loc[:, t + pd.DateOffset(days=i)].values


    # Sales feature last two year
    for i in range(-21, 21):
        X[f'sales_{i}_last_year'] = df_sales.loc[:, t - pd.DateOffset(days=365 + i)].values.ravel()
    """
    for i in [3, 7, 15]:
        last_year_idays_sales = df_sales.loc[:, t - pd.DateOffset(days=365 - 1): t - pd.DateOffset(days=365 - i)]

        X[f'last_year_{i}days_sales_diff_mean'] = last_year_idays_sales.diff(axis=1).mean(axis=1).values
        X[f'last_year_{i}days_sales_decay'] = (last_year_idays_sales * np.power(weight_factor, np.arange(
            last_year_idays_sales.shape[1]))).sum(axis=1).values
        X[f'last_year_{i}days_sales_mean'] = last_year_idays_sales.mean(axis=1).values
        X[f'last_year_{i}days_sales_median'] = last_year_idays_sales.median(axis=1).values
        X[f'last_year_{i}days_sales_min'] = last_year_idays_sales.min(axis=1).values
        X[f'last_year_{i}days_sales_max'] = last_year_idays_sales.max(axis=1).values
        X[f'last_year_{i}days_sales_std'] = last_year_idays_sales.std(axis=1).values
    """

    X['last_year_future_3days_sale_mean'] = df_promo.loc[:, t - pd.DateOffset(days=364): t - pd.DateOffset(days=362)].mean(axis=1).values
    X['last_year_future_7days_sale_mean'] = df_promo.loc[:, t - pd.DateOffset(days=364): t - pd.DateOffset(days=358)].mean(axis=1).values
    X['last_year_future_15days_sale_mean'] = df_promo.loc[:, t - pd.DateOffset(days=364): t - pd.DateOffset(days=350)].mean(axis=1).values

    X = pd.DataFrame(X)

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X