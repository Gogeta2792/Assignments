import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import inv

from sklearn.preprocessing import PolynomialFeatures


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0249011R(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # Step a, Split Iris dataset into training and test data sets
    
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = N)

    # Step b, Perform one-hot encoding
    
    onehot_encoder = OneHotEncoder(sparse=False)
    Ytr = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    Yts = onehot_encoder.transform(y_test.reshape(-1, 1))
    
    # Step c, Polynomial Regression
    
    Ptrain_list = [PolynomialFeatures(n).fit_transform(X_train) for n in range(1,9)]  # List of training polynomial matrices
    Ptest_list = [PolynomialFeatures(n).fit_transform(X_test) for n in range(1,9)]   # List of test polynomial matrices
    
    w_list = []
    


    lambda_value = 0.0001

    orders = range(1, 9)

    for P in Ptrain_list:
        # Check if the number of rows in the training polynomial matrix is less than or equal to the number of columns
        if P.shape[0] <= P.shape[1]:
            
            # Use the dual form of ridge regression, since rows <= columns
            w = P.T @ inv(P @ P.T + lambda_value * np.identity(P.shape[0])) @ Ytr
            w_list.append(w)
            
        else:
        
            # Use the primal form of ridge regression, since rows > columns
            w = inv(P.T @ P + lambda_value * np.identity(P.shape[1])) @ P.T @ Ytr
            w_list.append(w)
    
    error_train_array = np.zeros(8, dtype=int)
    error_test_array = np.zeros(8, dtype=int)
    
    for i in range(8):
        ytr_pred = Ptrain_list[i] @ w_list[i]
        ytr_class = np.argmax(ytr_pred, axis=1)
        error_train_array[i] = sum(y_train != ytr_class)
        
        yts_pred = Ptest_list[i] @ w_list[i]
        yts_class = np.argmax(yts_pred, axis=1)
        error_test_array[i] = sum(y_test != yts_class)
        
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
