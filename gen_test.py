"""
This script is for testing the functions defined in MatrixDecomp.py
with predetermined examples to make sure the results are as desired.
"""
import qoptics_matrices as qo
import numpy as np

dim = 4

print("Generating two random passive transformations:")
pas1 = qo.random_passive(dim)
print(pas1)
pas2 = qo.random_passive(dim)
print(pas2)
print("Check that both are valid passive transformations:", qo.check_valid_passive(pas1), qo.check_valid_passive(pas2))
print('\n')

print("Generating random squeeze:")
rsq = qo.random_squeezing(dim)
print(rsq)
print('\n')

print("Computing the resulting symplectic matrix:")
rand_sp = pas1 @ rsq @ pas2
print(rand_sp)
print("Check if valid symplectic matrix:", qo.check_valid_symplectic(rand_sp))
print('\n')

print("Generating covariance matrix of random thermal state:")
rand_therm = qo.random_thermal_state(dim)
print(rand_therm)
print('\n')

print("Computing the resulting Gaussian state covariance matrix:")
rand_gaus = rand_sp @ rand_therm @ rand_sp.T
print(rand_gaus)
print("Check if valid covariance matrix (for Williamson decomposition):", qo.check_valid_covariance(rand_gaus))
print('\n')

print("Generating a second symplectic matrix:")
rand_sp2 = qo.random_symplectic(dim)
print(rand_sp2)
print("Check if valid symplectic matrix:", qo.check_valid_symplectic(rand_sp))
