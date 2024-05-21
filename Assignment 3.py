import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0249011R(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    
     #Initialization values
    a = 1.5
    b = 0.3
    c = 1
    d = 2
    
    #Creating numpy arrays
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    
    
    #Part a
    a_out[0] = a - learning_rate * (5 * a ** 4)
    f1_out[0] = a_out[0] ** 5

    #Part b
    b_out[0] = b - learning_rate * (2 * np.cos(b) * np.sin(b))
    f2_out[0] = np.sin(b_out[0]) ** 2

    #Part c
    c_out[0] = c - learning_rate * (3 * c ** 2)
    d_out[0] = d - learning_rate * (2 * d * np.sin(d) + d ** 2 * np.cos(d))
    f3_out[0] = (c_out[0] ** 3) + (d_out[0] ** 2 * np.sin(d_out[0]))
            
    for i in range (1, num_iters):
        #Part a
        a_out[i] = a_out[i-1] - learning_rate * (5 * a_out[i-1] ** 4)
        f1_out[i] = a_out[i] ** 5

        #Part b
        b_out[i] = b_out[i-1] - learning_rate * (2 * np.cos(b_out[i-1]) * np.sin(b_out[i-1]))
        f2_out[i] = np.sin(b_out[i]) ** 2

        #Part c
        c_out[i] = c_out[i-1] - learning_rate * (3* c_out[i-1] ** 2)
        d_out[i] = d_out[i-1] - learning_rate * (2 * d_out[i-1] * np.sin(d_out[i-1]) + d_out[i-1] ** 2 * np.cos(d_out[i-1]))
        f3_out[i] = (c_out[i] ** 3) + (d_out[i] ** 2 * np.sin(d_out[i]))
            
    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 