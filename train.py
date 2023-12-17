import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import PReLU
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(16))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Dense(1))

    return model


def train_gbm(X_train, y_train, X_val, y_val, X_test, params):
    models = []
    val_pred = []
    test_pred = []

    for i in range(16):
        print('-' * 50)
        print(f'Predict Day {i + 1}')
        print('-' * 50)

        dtrain = lgb.Dataset(
            X_train,
            label=y_train[:, i]
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val[:, i],
            reference=dtrain,
        )
        bst = lgb.train(
            params['model'],
            dtrain,
            num_boost_round=params['boost_round'],
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params['early_stopping_rounds']),
                lgb.log_evaluation(period=50)
            ]
        )

        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))

        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or params['boost_round']))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or params['boost_round']))
        models.append(bst)

    return models, val_pred, test_pred


def train_general_gbm(X_train, y_train, X_val, y_val, X_test, params):
    models = []
    val_pred = []
    test_pred = []

    dtrain = lgb.Dataset(
        X_train,
        label=y_train
        )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
    )
    bst = lgb.train(
        params['model'],
        dtrain,
        num_boost_round=params['boost_round'],
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=params['early_stopping_rounds']),
            lgb.log_evaluation(period=50)
        ]
    )
    """
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    """
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or params['boost_round']))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or params['boost_round']))
    models.append(bst)

    return models, val_pred, test_pred


def train_nn(X_train, y_train, X_val, y_val, X_test, epoch, input_shape):
    models = []
    val_pred = []
    test_pred = []

    for i in range(16):
        print('-' * 50)
        print(f'Predict Day {i + 1}')
        print('-' * 50)

        y = y_train[:, i]
        y_mean = y.mean()
        xv = X_val
        yv = y_val[:, i]
        model = build_model(input_shape)
        opt = optimizers.Adam(lr=0.001)
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        ]

        model.fit(X_train,
                  y - y_mean,
                  batch_size=512,
                  epochs=epoch,
                  verbose=2,
                  validation_data=(xv, yv - y_mean),
                  callbacks=callbacks)
        val_pred.append(model.predict(X_val) + y_mean)
        test_pred.append(model.predict(X_test) + y_mean)

    return models, val_pred, test_pred
