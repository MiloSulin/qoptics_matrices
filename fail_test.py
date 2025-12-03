"""
Testing the generation of larger symplectic matrices.
"""
from collections import Counter
import numpy as np
import qoptics_matrices as qo

true_or_false = []
for i in range(10000):
    true_or_false.append(qo.check_valid_symplectic(qo.random_symplectic(32)))
print(Counter(true_or_false))

true_or_false = []
for i in range(10000):
    true_or_false.append(qo.check_valid_symplectic(qo.random_symplectic(64)))
print(Counter(true_or_false))

true_or_false = []
fixes = 0
for i in range(10000):
    matrix = qo.random_symplectic(64)
    result = qo.check_valid_symplectic(matrix)
    if result is False:
        nu_matrix = qo.fix_symplectic(matrix)
        result = qo.check_valid_symplectic(nu_matrix)
        if result:
            fixes += 1
    true_or_false.append(result)
print(Counter(true_or_false))
print(fixes)
