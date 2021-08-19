import numpy as np
import math
from typing import List, Union
import matplotlib.pyplot as plt

def Jacobi_rotation_method(A: np.array, eps: float) -> List[Union[np.array, np.array, int]]:
    '''Get eigenvectors and eigenvalues of the matrix A with precision eps'''

    iter_count = 0
    A = np.array(A)
    n = len(A)

    u, u_k, a_k = np.eye(n), np.eye(n), np.copy(A)

    phi_k = 0
    max_elem_pos = [0, 0]
    max_elem = 0

    while math.sqrt(sum([a_k[i, j]**2 for i in range(n) for j in range(n) if i != j])) > eps:
        max_elem = -1
        for i in range(n):
            for j in range(i + 1, n):
                if i != j and abs(a_k[i, j]) > max_elem:
                    max_elem = abs(a_k[i, j])
                    max_elem_pos = [i, j]

        if a_k[max_elem_pos[0], max_elem_pos[0]] == a_k[max_elem_pos[1], max_elem_pos[1]]:
            phi_k = math.pi / 4
        else:
            phi_k = 0.5 * math.atan(2 * a_k[max_elem_pos[0], max_elem_pos[1]] / (a_k[max_elem_pos[0], max_elem_pos[0]] - a_k[max_elem_pos[1], max_elem_pos[1]]))

        u_k[max_elem_pos[0], max_elem_pos[0]] = math.cos(phi_k)
        u_k[max_elem_pos[1], max_elem_pos[1]] = math.cos(phi_k)
        u_k[max_elem_pos[0], max_elem_pos[1]] = math.sin(phi_k)
        u_k[max_elem_pos[1], max_elem_pos[0]] = -math.sin(phi_k)

        a_k = u_k @ a_k
        u_k = u_k.T
        a_k = a_k @ u_k
        u = u @ u_k
        u_k = np.eye(n)
        iter_count += 1

    return [a_k, u, iter_count]


def SVD(A: np.array, eps: float) -> List[np.array]:
    '''Singular value decomposition of matrix A with precision = eps.
    A = U * S * V^T
    Returns a list of 3 matrices U, S, V^T.
    '''

    AT_A = np.dot(A.T, A)

    # get the diagonal matrix with eigenvalues on its diagonal
    # and a matrix which has eigenvectors as its columns
    AT_A_eigenvalues, AT_A_eigenvectors, _ = Jacobi_rotation_method(AT_A, eps)

    # Construct the diagonal matrix, because Jacobi rotation method is not completely precise
    n = len(AT_A_eigenvalues)
    AT_A_eigenvalues = [[AT_A_eigenvalues[i][j] if i == j else 0 for j in range(n)] for i in range(n)]

    # In case A is singular, we need to get 0 instead of a really small number of its diagonal
    AT_A_eigenvalues = [[AT_A_eigenvalues[i][j] if abs(AT_A_eigenvalues[i][j]) > 1e-8 else 0 for j in range(n)] for i in range(n)]

    S = np.sqrt(AT_A_eigenvalues)
    V = np.copy(AT_A_eigenvectors)

    # U = A * V * S^(-1)
    # S^(-1) = S^+ is the pseudoinverse in case A is singular (non-invertible)
    S_inverse = np.array([[1/S[i][j] if (i == j and S[i][j] != 0) else 0 for j in range(len(S))] for i in range(len(S))])
    U = np.linalg.multi_dot([A, V, S_inverse])

    return [U, S, V.T]


if __name__ == '__main__':

    from skimage import io
    img = io.imread('photo.png', as_gray=True)
    arr = np.array(img)

    U, S, VT = SVD(arr, 0.0001)

    compressed_image = np.zeros(arr.shape)
    max_singular_values = 100

    for i in range(0, max_singular_values):
        compressed_image += S[i] * np.outer(U[:, i], VT[i, :])

    fig = plt.figure(dpi=180)
    plt.imshow(compressed_image, cmap='gray', interpolation=None)
