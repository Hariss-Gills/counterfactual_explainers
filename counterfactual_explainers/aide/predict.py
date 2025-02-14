import keras
import numpy as np
import tensorflow as tf


def predict_no_reshape(model, x):
    y = model(x)
    print("y", y)
    return y


# def predict_single(model, x):  # get y prediction from single array of X
#     # make 1D x into 3D (None,None,1)
#     x = x.reshape(1, 1, -1)
#     y = model(x)
#     # get 3D tensor back to 1D y array
#     y = np.array(y[0][0][0])
#
#     return y  # 1D array out


def predict_single(model, x):  # get y prediction from single array of X
    # Reshape 1D x into 2D (batch_size=1, features)
    x = x.reshape(1, -1)
    y = model(x)
    # Extract the scalar prediction from the 2D output array
    y = np.array(y[0][0])  # Adjust based on model's output shape

    return y  # Return as a scalar or 1D array


# def predict_batch(model, X):  # get y prediction from single array of X
#     # make 2D x into 3D (None,None,1)
#     X_dim_1 = 1
#     X_dim_2 = X.shape[0]
#     X_dim_3 = X.shape[1]
#
#     X = X.reshape(X_dim_1, X_dim_2, X_dim_3)
#     y = model.predict(X, batch_size=None, verbose=None)
#     # get 3D tensor back to 1D y array
#     y = y[0].reshape(-1)
#     return y  # output 1D  array
#


def predict_batch(model, X):
    # Ensure X is 2D (batch_size, 103)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Fix single-sample input
    y = model.predict(X, batch_size=None, verbose=None)
    return y  # Shape (batch_size, 103)


def predict_1D(model, x):  # imput is 1D array not changed internally
    y = model(x)
    return y


def predict_2D(model, x):  # input is made a 2D array
    x = x.reshape(1, -1)
    y = model(x)
    y = np.array(y[0][0])
    return y
