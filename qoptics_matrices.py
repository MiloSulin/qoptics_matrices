"""
This module includes functions for performing the Williamson and
Bloch-Messiah/Euler decompositions for symmetric positive definite
and symplectic matrices respectively.
Additionally functions for generating such matrices randomly are
also included.
"""
import numpy as np
import scipy as sp


# Function for generating an arbitrary symplectic matrix.
def random_symplectic(dim : int):
    """
    This function generates a symplectic matrix with of shape dim x dim.
    We achieve this by first generating a random element of the corresponding symplectic
    Lie algebra and then exponentiating said element.
    INPUT:
    dim - integer which defines the shape of the output matrix. Has to be divisible by 2.
    OUTPUT:
    random_sym - The generated random symplectic matrix.
    """
    if dim % 2 != 0:
        print("Invalid argument! Dimension must be divisible by two (2)")
        return np.empty(1)
    dim_2 = dim // 2

    # The Lie algebra element is of the form [[A, B],[C, D]], where A,B,C, and D are square matrices of
    # shape dim_2xdim_2. Additionally A^t = -D, and C and B are symmetric.
    random_gen = np.random.default_rng()
    random_square_matrix = random_gen.normal(size=(dim_2, dim_2))
    b_upper_triangle = np.triu(random_square_matrix, 1)
    b_diagonal = np.diag(np.diagonal(random_square_matrix))
    c_lower_triangle = np.tril(random_square_matrix, -1)
    c_diagonal = np.diag(random_gen.normal(size=dim_2))
    # We construct two symmetric matrices from these components
    b_full = b_diagonal + b_upper_triangle +b_upper_triangle.T
    c_full = c_diagonal + c_lower_triangle +c_lower_triangle.T

    # Generate a random square matrix A (and D since D = -A^t)
    a_matrix = random_gen.normal(size=(dim_2, dim_2))
    d_matrix = -a_matrix.T

    # Construct the element of the symplectic Lie algebra
    row_one = np.concatenate([a_matrix, b_full], axis=1)
    row_two = np.concatenate([c_full, d_matrix], axis=1)
    lie_algebra_matrix = np.concatenate([row_one, row_two], axis=0)

    return sp.linalg.expm(lie_algebra_matrix)

def fix_symplectic(symp_m : np.ndarray):
    s_form = get_symplectic_form(len(symp_m))
    id_mat = np.identity(len(symp_m))
    v_matrix = s_form @ (id_mat - symp_m) @ np.linalg.inv(id_mat + symp_m)
    w_matrix = (v_matrix + v_matrix.T) / 2
    if np.linalg.det(id_mat - (s_form @ w_matrix)) != 0:
        m_matrix = (id_mat + (s_form @ w_matrix) ) @ np.linalg.inv(id_mat - (s_form @ w_matrix) )
        return m_matrix
    v_matrix = s_form @ (id_mat + symp_m) @ np.linalg.inv(id_mat - symp_m)
    w_matrix = (v_matrix + v_matrix.T) / 2
    m_matrix = (id_mat + (s_form @ w_matrix) ) @ np.linalg.inv(id_mat - (s_form @ w_matrix) )
    return m_matrix

def random_passive(dim : int):
    """
    This function generates a matrix with shape dim x dim which corresponds to some
    passive phase space transformation. This matrix is both orthogonal and symplectic.
    We achieve this by first generating a random element of the corresponding
    Lie algebra and then exponentiating said element. This function is
    only for even sized matrices as it is meant for constructing operations in quantum
    optics phase space.
    INPUT:
    dim - integer which defines the shape of the output matrix. Has to be divisible by 2.
    OUTPUT:
    random_ort - The generated random symplectic matrix.
    """
    if dim % 2 != 0:
        print("Invalid argument! Dimension must be divisible by two (2)")
        return np.empty(1)
    dim_2 = dim // 2

    # The Lie algebra element is of the form [[A,B],[C,D]], where A,B,C,D are square matrices of
    # shape dim_2xdim_2. Additionally C = -B, B is symmetric and A is antisymmetric and
    # D = A
    random_gen = np.random.default_rng()
    random_square_matrix = random_gen.normal(size=(dim_2, dim_2))
    a_upper_triangle = np.triu(random_square_matrix, 1)
    b_diagonal = np.diag(np.diagonal(random_square_matrix))
    b_lower_triangle = np.tril(random_square_matrix, -1)
    # We construct two antisymmetric matrices from these components
    a_full = a_upper_triangle -a_upper_triangle.T
    d_full = a_full

    b_matrix = b_diagonal + b_lower_triangle +b_lower_triangle.T
    c_matrix = -b_matrix

    # Construct the element of the symplectic Lie algebra
    row_one = np.concatenate([a_full, b_matrix], axis=1)
    row_two = np.concatenate([c_matrix, d_full], axis=1)
    lie_algebra_matrix = np.concatenate([row_one, row_two], axis=0)

    return sp.linalg.expm(lie_algebra_matrix)

def random_squeezing(dim : int):
    """
    This function generates a random diagonal matrix corresponding to
    singlemode squeezing of a phase space of size dim. We generate
    dim // 2 number of values from a distribution, take their negatives
    and then shuffle these pairs to randomize which quadrature is squeezed.
    """
    if dim % 2 != 0:
        print("Invalid argument! Dimension must be divisible by two (2)")
        return np.empty(1)
    dim_2 = dim // 2
    random_gen = np.random.default_rng()
    array_of_values = np.abs(random_gen.normal(size=dim_2))
    array_of_inv = -array_of_values
    diag_array = np.concatenate([array_of_values, array_of_inv])
    return np.diag(np.exp(diag_array))

def random_thermal_state(dim : int):
    """
    Generates the covariance matrix of a random thermal state.
    """
    dim_2 = dim // 2
    random_gen = np.random.default_rng()
    symp_eigen_values = random_gen.uniform(low=0.5, high=2, size=dim_2)
    symp_eigen_values = np.concatenate([symp_eigen_values, symp_eigen_values])
    # form the positive definite covaraince matrix based on Williamson decomposition
    diag_matrix = np.diag(symp_eigen_values)
    return diag_matrix

def random_covariance(dim : int):
    """
    This function generates a symmetric positive definite matrix that is valid
    as the covariance matrix of a quantum optics system. We perform this by first
    generating a random symplectic matrix of size dim and generating the diagonal
    decomposition values corresponding to the thermal state properties of the modes.
    INPUT:
    dim - integer which defines the shape of the output matrix. Has to be divisible by 2.
    OUTPUT:
    random_cov - The generated random covariance matrix.
    """
    if dim % 2 != 0:
        print("Invalid argument! Dimension must be divisible by two (2)")
        return np.empty(1)
    dim_2 = dim // 2

    # first generate symplectic matrix
    symplectic_matrix = random_symplectic(dim)
    # then generate the necessary symplectic eigenvalues
    random_gen = np.random.default_rng()
    symp_eigen_values = random_gen.uniform(low=0.5, high=2, size=dim_2)
    symp_eigen_values = np.concatenate([symp_eigen_values, symp_eigen_values])
    # form the positive definite covaraince matrix based on Williamson decomposition
    diag_matrix = np.diag(symp_eigen_values)
    random_cov = symplectic_matrix @ diag_matrix @ symplectic_matrix.T

    return random_cov

def check_valid_covariance(input_matrix : np.ndarray):
    """
    This function checks if the argument square matrix is a positive definite one
    and symmetric. The positive definiteness is checked by attempting the Cholesky decomposition
    which will fail if this requirement is not fulfilled. Technically a covariance matrix can be
    only positive semidefinite but in the context of Williamson decomposition full definiteness
    is required.
    Use of Cholesky decomposition in this manner was learned from:
    https://stackoverflow.com/questions/16266720/find-out-if-a-matrix-is-positive-definite-with-numpy
    INPUT:
    input_matrix - Input argument
    OUTPUT:
    bool_result - True or False depending on if argument is a valid covariance matrix
    """
    try:
        _ = np.linalg.cholesky(input_matrix)
    except np.linalg.LinAlgError:
        return False
    if np.allclose(input_matrix, input_matrix.T):
        # check that matrix is also symmetric
        return True
    return False

def get_symplectic_form(dim : int):
    """
    This function constructs the symplectic form [[null, identity], [-identity, null]] of size
    dim x dim.
    """
    dim_2 = dim // 2
    id_matrix = np.identity(dim_2)
    null_matrix = np.zeros(shape=(dim_2, dim_2))
    row_one = np.concatenate([null_matrix, id_matrix], axis=1)
    row_two = np.concatenate([-id_matrix, null_matrix], axis=1)
    symp_form = np.concatenate([row_one, row_two], axis=0)
    return symp_form

def get_orthogonal_form(dim : int):
    """
    This function constructs the orthogonal form [[null, identity], [identity, null]] of size
    dim x dim.
    """
    dim_2 = dim // 2
    id_matrix = np.identity(dim_2)
    null_matrix = np.zeros(shape=(dim_2, dim_2))
    row_one = np.concatenate([null_matrix, id_matrix], axis=1)
    row_two = np.concatenate([id_matrix, null_matrix], axis=1)
    orto_form = np.concatenate([row_one, row_two], axis=0)
    return orto_form

def kron_delta(i,j):
    """
    Function for Kronecker delta function.
    """
    if i == j:
        return 1
    return 0

def perm_matrix_element(i,j, dim_2):
    """
    Function for calculating the element of symplectic permutation matrix
    at indices i and j. Argument dim_2 describes the amount of modes in the
    quantum optics system, in other words the dimension of the symplectic matrix
    divided by 2.
    """
    input_i = i+1
    input_j = j+1
    return kron_delta(input_j, 2*input_i-1) + kron_delta(input_j +2*dim_2, 2*input_i)

v_perm_matrix_element = np.vectorize(perm_matrix_element)

def get_symplectic_perm(dim : int):
    """
    This function computes the permutation matrix for changing between the 'standard'
    and block-diagonal symplectic matrices.
    INPUT:
    dim - Dimension of the output dimension, has to be divisible by 2
    OUTPUT:
    perm_matrix - Permutation matrix
    """
    if dim % 2 != 0:
        print("Invalid argument! Dimension must be divisible by two (2)")
        return np.empty(1)
    dim_2 = dim // 2
    # construct permutation matrix
    perm_matrix = np.fromfunction(v_perm_matrix_element, shape=(dim, dim), dim_2 = dim_2)
    return perm_matrix


def check_valid_symplectic(input_matrix : np.ndarray):
    """
    This function checks if input argument is a symplectic matrix.
    That is; let S be the input then if S is symplectic
    -> S @ symp_form @ S.T = symp_form.
    Because of floating point errors this will almost never be exactly the case
    we will use the numpy function allclose to see if the resulting matrix is
    close enough to the original symplectic form. The default tolerance is around
    1.0e(-8).
    INPUT:
    input_matrix - Matrix to be checked.
    OUTPUT:
    bool_result - Returns True or False depending on if argument is a symplectic matrix
    """
    dim = len(input_matrix)
    if dim % 2 != 0:
        # input matrix is not of even dimension
        return False
    # we construct the symplectic form of appropriate size
    symp_form = get_symplectic_form(dim)

    # check that matrix fulfills the symplectic condition
    matrix_product = input_matrix @ symp_form @ input_matrix.T
    bool_result = np.allclose(matrix_product, symp_form)

    return bool_result

def check_valid_orthogonal(input_matrix : np.ndarray):
    """
    This function checks if input argument is a symplectic matrix.
    That is; let S be the input then if S is symplectic
    -> S @ symp_form @ S.T = symp_form.
    Because of floating point errors this will almost never be exactly the case
    we will use the numpy function allclose to see if the resulting matrix is
    close enough to the original symplectic form. The default tolerance is around
    1.0e(-8).
    INPUT:
    input_matrix - Matrix to be checked.
    OUTPUT:
    bool_result - Returns True or False depending on if argument is a symplectic matrix
    """
    dim = len(input_matrix)
    if dim % 2 != 0:
        # input matrix is not of even dimension
        return False
    # we construct the symplectic form of appropriate size
    # orto_form = get_orthogonal_form(dim)
    matrix_product1 = input_matrix @ input_matrix.T
    matrix_product2 = input_matrix.T @ input_matrix
    # check that matrix fulfills the symplectic condition
    # matrix_product = input_matrix @ orto_form @ input_matrix.T
    bool_result1 = np.allclose(matrix_product1, np.identity(dim))
    bool_result2 = np.allclose(matrix_product2, np.identity(dim))

    return bool_result1 and bool_result2

def check_valid_passive(input_matrix : np.ndarray):
    """
    This function checks if the given input is a valid passive transformation
    matrix. That is, it checks that input is both symplectic and orthogonal.
    """
    bool_result1 = check_valid_symplectic(input_matrix)
    bool_result2 = check_valid_orthogonal(input_matrix)
    return bool_result1 and bool_result2


def taka_auto_decomposition(input_matrix : np.ndarray):
    """
    This function calculates the Takagi/Autonne decomposition of the input matrix
    using the numpy function for the singular value decomposition.
    INPUT:
    input_matrix - Matrix to be decomposed.
    OUTPUT:
    s_matrix - Diagonal matrix of eigen values
    w_matrix - The unitary Takagi/Autonne matrix
    """
    u_matrix, s_array, vh_matrix = np.linalg.svd(input_matrix)
    s_matrix = np.diag(s_array)

    # calulate Takagi/Autonne unitary
    uv_matrix = u_matrix.T @ np.conjugate(vh_matrix.T)
    sqrt_matrix = sp.linalg.sqrtm(uv_matrix.conjugate())
    w_matrix = u_matrix @ sqrt_matrix

    return (s_matrix, w_matrix)

def bme_decomposition(input_matrix : np.ndarray):
    """
    This function computes the Bloch-Messiah/Euler decomposition of the input matrix
    using the polar decomposition from scipy and the Takagi/Autonne decomposition
    function defined above. First we check that the given argument is a valid
    symplectic matrix using check_valid_symplectic.
    Note that we only work in the basis corresponding to the symplectic form
    [[null, identity], [-identity, null]] for now.
    INPUT:
    input_matrix - Matrix to decompose, has to be symplectic matrix
    OUTPUT:
    o_matrix1 - First orthogonal matrix of the decomposition
    d_matrix - Diagonal matrix of the decomposition
    o_matrix2 - Second orthogonal matrix of the decomposition
    """
    is_symplectic = check_valid_symplectic(input_matrix)
    if is_symplectic is False:
        print("Invalid argument! Input should be a symplectic matrix!")
        return (np.empty(1), np.empty(1), np.empty(1))

    # If argument isn't invalid, first compute the polar decomposition
    u_matrix, p_matrix = sp.linalg.polar(input_matrix, side='left')
    # take the blocks A,B,C from p_matrix = [[A,B], [B^t, C]]
    dim_2 = len(p_matrix) // 2
    a_matrix = p_matrix[:dim_2, :dim_2]
    b_matrix = p_matrix[:dim_2, dim_2:]
    c_matrix = p_matrix[dim_2:, dim_2:]
    # obtain matrix that is to be takagi/autonne decomposed
    m_matrix = (1/2)*( a_matrix -c_matrix +(b_matrix + b_matrix.T)*1j )
    # taka/auto decomp
    s_matrix, w_matrix = taka_auto_decomposition(m_matrix)
    # get imaginary and real parts of the unitary w_matrix
    w_imaginary = w_matrix.imag
    w_real = w_matrix.real
    # construct first orthogonal matrix
    row_one = np.concatenate([w_real, -w_imaginary], axis=1)
    row_two = np.concatenate([w_imaginary, w_real], axis=1)
    o_matrix1 = np.concatenate([row_one, row_two], axis=0)
    #construct second orthonormal matrix
    o_matrix2 = o_matrix1.T @ u_matrix
    # construct diagonal matrix
    gamma_m = s_matrix + sp.linalg.sqrtm( np.identity(dim_2) + np.linalg.matrix_power(s_matrix, 2))
    gamma_array = np.diagonal(gamma_m)
    gamma_inverse = 1 / gamma_array
    diag_array = np.concatenate([gamma_array, gamma_inverse])
    d_matrix = np.diag(diag_array)

    return (o_matrix1, d_matrix, o_matrix2)

def bool_to_matrix(input_val : bool):
    """
    This function returns a matrix corresponding to the appropriate permutation
    based on the input which tells if positive Schur decomp value was above diagonal
    or not.
    """
    if input_val:
        return np.array([[1,0],[0,1]])
    return np.array([[0,1],[1,0]])

v_bool_to_matrix = np.vectorize(bool_to_matrix, otypes=[np.ndarray])

def insert_in_matrix(index : int, sq_matrix, output_m):
    """
    inserts the square matrix along the diagonal of output_m at given index.
    """
    offset = index*len(sq_matrix)
    output_m[offset:offset+len(sq_matrix), offset:offset+len(sq_matrix)] = sq_matrix

v_insert_in_matrix = np.vectorize(insert_in_matrix, excluded={2, 'output_m'})

def simple_direct_sum(matrix_array):
    """
    This function constructs a simple direct sum of the given
    input matrices which are assumed to be square and all the same shape.
    """
    amount = matrix_array.shape[0]
    size = len(matrix_array[0])
    sum_size = amount*size
    output_m = np.zeros(shape=(sum_size, sum_size))
    indices = np.arange(amount)
    # attempt to perform direct summation
    v_insert_in_matrix(indices, matrix_array, output_m)
    return output_m

def schur_perm_matrix(b_diag_matrix : np.ndarray):
    """
    This function constructs the matrix which permutes the block-diagonal Schur
    matrix so that positive values are above the diagonal and negative values
    below the diagonal.
    INPUT:
    b_diag_matrix - Block-diagonal matrix from a Schur decomposition
    OUTPUT:
    perm_matrix - Matrix that will can be used to do the desired permutation
    """
    # begin by taking the upper and lower diagonal as arrays
    upper_diagonal = np.diagonal(b_diag_matrix, 1)[::2]
    lower_diagonal = np.diagonal(b_diag_matrix, -1)[::2]
    bool_array = upper_diagonal > lower_diagonal # find indices where pos value is above diagonal
    # pick which matrix for what index based on the above array of booleans
    matrix_array = v_bool_to_matrix(bool_array)
    perm_matrix = simple_direct_sum(matrix_array)

    return perm_matrix

def williamson_decomposition(input_matrix : np.ndarray):
    """
    This function computes the Williamson decomposition for the given input.
    For the matrix M we find the decomposition
    M = S @ T @ S^t, where S is symplectic and T is diagonal and direct sum of
    some matrix t with itself (t + t).
    To compute the Williamson decomposition the Schur decomposition function
    profided by Scipy is used as well as the symplectic permutation matrix
    function defined above.
    INPUT:
    input_matrix - Matrix to be decomposed.
    OUTPUT:
    d_matrix - Diagonal matrix of the symplectic eigenvalues
    s_matrix - The symplectic matrix of the decomposition
    """
    # first check that input is valid as a covariance matrix (symmetric and positive definite)
    is_valid_cov = check_valid_covariance(input_matrix)
    if is_valid_cov is False:
        print("Invalid argument! Input matrix should be symmetric and poisitive definite to perform Williamson decomposition")
        return (np.empty(1), np.empty(1))
    # If valid, perform decomposition
    # begin with constructing antisymmetric matrix
    cov_sqrt = sp.linalg.sqrtm(input_matrix)
    dim = len(input_matrix)
    symp_form = get_symplectic_form(dim)
    cov_sqrt_inv = np.linalg.inv(cov_sqrt)
    anti_sym_matrix = cov_sqrt_inv @ symp_form @ cov_sqrt_inv
    # compute Schur decomposition
    t_matrix, z_matrix = sp.linalg.schur(anti_sym_matrix)
    # get schur and symplectic permutation matrices
    schur_perm = schur_perm_matrix(t_matrix)
    symp_perm = get_symplectic_perm(len(input_matrix))
    #get positive eigenvalues from Schur decomposition
    t_arranged = schur_perm.T @ t_matrix @ schur_perm
    eigenv_array = np.diagonal(t_arranged, 1)[::2]

    # construct ingredients for Symplectic matrix
    eigenv_arr_full = np.concatenate([eigenv_array, eigenv_array])
    eigen_matrix = np.diag(eigenv_arr_full)
    eigen_m_sqrt = sp.linalg.sqrtm(eigen_matrix)
    # finally we get
    s_matrix = cov_sqrt @ z_matrix @ schur_perm @ symp_perm @ eigen_m_sqrt
    # construct d_matrix
    d_matrix = np.linalg.inv(eigen_matrix)

    return (d_matrix, s_matrix)
