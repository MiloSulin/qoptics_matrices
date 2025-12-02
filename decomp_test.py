"""
This script tests the Bloch-Messiah/Euler and Williamson
decompositions for predetermined 4x4 symplectic and covariance
matrices.
"""
import numpy as np
import qoptics_matrices as qo

print("Testing Bloch-Messiah/Euler decomposition...")
passive_transformation1 = np.array([[ 0.06089432999, -0.83098399999,  0.20564336999, -0.51329164999],
                                    [ 0.93043927999,  0.10881185999, -0.29712868999, -0.18481691999],
                                    [-0.20564336999,  0.51329164999,  0.06089432999, -0.83098399999],
                                    [ 0.29712868999,  0.18481691999,  0.93043927999,  0.10881185999]])

passive_transformation2 = np.array([[ 0.7523676205,  0.4707662436, -0.1338549511, -0.4409137778],
                                    [ 0.1237977292,  0.2859068792, -0.6330086397,  0.7086828916],
                                    [ 0.1338549511,  0.4409137778,  0.7523676205,  0.4707662436],
                                    [ 0.6330086397, -0.7086828916,  0.1237977292,  0.2859068792]])


squeezing_matrix = np.array([[2.39955992, 0.0,         0.0,         0.0,        ],
                             [0.0,        2.68969762,  0.0,         0.0,        ],
                             [0.0,        0.0,         0.41674308,  0.0,        ],
                             [0.0,        0.0,         0.0,         0.371789   ]])

# Compute the resulting symplectic matrix:
symplectic1 = passive_transformation1 @ squeezing_matrix @ passive_transformation2
# if the following function succeeds this matrix was indeed symplectic
d_pas1, d_sqz, d_pas2 = qo.bme_decomposition(symplectic1)
print("Matrix to be decomposed:")
print(symplectic1, '\n')
print("Printing the BME decomposition:")
print("Orthogonal/passive matrix 1:", '\n', d_pas1)
print("Diagonal squeezing matrix:", '\n', d_sqz)
print("Orthogonal/passive matrix 2:", '\n', d_pas2, '\n')


thermal_covariance = np.array([[1.56929955, 0.0,        0.0,        0.0        ],
                               [0.0,        1.75082925, 0.0,        0.0        ],
                               [0.0,        0.0,        1.56929955, 0.0        ],
                               [0.0,        0.0,        0.0,        1.75082925 ]])
print("Testing Williamson decomposition...")
print("Original thermal covariance:", '\n', thermal_covariance)
print("Computing Gaussian state covariance matrix:")
gaus_cov = symplectic1 @ thermal_covariance @ symplectic1.T
print(gaus_cov, '\n')
print("Printing Williamson decomposition:")
d_therm, d_symp = qo.williamson_decomposition(gaus_cov)
print("Diagonal thermal matrix:", '\n', d_therm)
print("Symplectic matrix:", '\n', d_symp)
