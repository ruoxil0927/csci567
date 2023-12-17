import numpy as np
import pandas as pd


def make_submission(df_train, df_test, df_sales, test_pred, name, zero_forecast=True):
    y_test = np.array(test_pred).transpose().squeeze()
    # y_test = test_pred
    df_preds = pd.DataFrame(
        y_test, index=df_sales.index,
        columns=pd.date_range("2017-08-16", periods=16).to_period('D')
    ).stack().to_frame("sales")
    df_preds.index.set_names(['store_nbr', 'family', 'date'], inplace=True)

    if zero_forecast:
        c = df_train.groupby(["store_nbr", "family"]).tail(15).groupby(["store_nbr", "family"]).sales.sum().reset_index()
        c = c[c.sales == 0].drop("sales", axis=1)
        # c = c[c.family != 31]
        for index, row in c.iterrows():
            df_preds.loc[(row['store_nbr'], row['family'])] = 0

    submission = df_test[["id"]].join(df_preds, how="left").fillna(0.)
    # Map value back
    submission['sales'] = np.clip(np.expm1(submission['sales']), 0, None)
    submission = submission.sort_values(by='id')
    submission.to_csv(name, float_format='%.6f', index=None)
