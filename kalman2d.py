import sys
import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 5):
        print ("Four arguments required: python kalman2d.py [datafile] [x1] [x2] [lambda]")
        exit()
    
    filename = sys.argv[1]
    x10 = float(sys.argv[2])
    x20 = float(sys.argv[3])
    scaler = float(sys.argv[4])

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    data = []
    for line in range(0, len(lines)):
        data.append(list(map(float, lines[line].split(' '))))

    # Print out the data
    print ("The input data points in the format of 'k [u1, u2, z1, z2]', are:")
    for it in range(0, len(data)):
        print (str(it + 1) + ": ", end='')
        print (*data[it])

    k = len(data)

    # Take input and create u (control input) and z (observation) matrix arrays (to be used in filter)
    u = []
    z = []

    for i in range(k):
        u.append(np.matrix([[data[i][0]], [data[i][1]]]))
        z.append(np.matrix([[data[i][2]], [data[i][3]]]))

    # Define constants and given variables
    Q = np.matrix([[10**(-4), 2 * 10**(-5)], [2 * 10**(-5), 10**(-4)]])  # process covariance
    R = np.matrix([[10**(-2), 5 * 10**(-3)], [5 * 10**(-3), 2 * 10**(-2)]])  # measurement covariance
    x_0 = np.matrix([[x10], [x20]]) # Initial predicted state
    P_0 = scaler * np.identity(2)   # Initial predicted error covariance

    # Initialize prediction/update variables
    x_hat_a_priori = [None]*(k+1)   # a priori predicted state
    x_hat_a_priori[0] = x_0 + u[0]
    P_a_priori = [None]*(k+1)       # a priori predicted error covariance
    P_a_priori[0] = P_0 + Q

    x_hat = [None]*(k)  # predicted state
    K = [None]*(k)      # Kalman gain factor
    P = [None]*(k)     # error covariance

    # Kalman Filter implementation
    for i in range(k):
        # Measurement Update
        K[i] = np.dot(P_a_priori[i], np.linalg.inv(P_a_priori[i] + R))
        x_hat[i] = x_hat_a_priori[i] + np.dot(K[i], z[i] - x_hat_a_priori[i])
        P[i] = np.dot(np.identity(2) - K[i], P_a_priori[i])
        
        # Time Update
        x_hat_a_priori[i+1] = x_hat[i] + u[i]
        P_a_priori[i+1] = P[i] + Q
    
    # Plot x(1,k), x(2,k) values against z(1,k),z(2,k)
    x_1k = []
    x_2k = []
    z_1k = []
    z_2k = []

    for i in range(k):
        x_1k.append(x_hat[i].item((0,0)))
        x_2k.append(x_hat[i].item((1,0))) 
        z_1k.append(data[i][2])
        z_2k.append(data[i][3])

    plt.plot(x_1k, x_2k,'-r', marker = 'o', label = 'Predicted')
    plt.plot(z_1k, z_2k,'-b', marker = 'o', label = 'Observed')
    plt.legend(loc = 'upper right')
    plt.show()