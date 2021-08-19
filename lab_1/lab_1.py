from typing import Sequence
import numpy as np
import math

class Matrix():
    def __init__(self, matrix):
        self.m = np.array(matrix, dtype='float64')

    def __str__(self):
        return str(self.m)
    
    def multiply(self, rhs: 'Matrix') -> 'Matrix':
        '''Matrix multipication using numpy.'''
        return Matrix(self.m.dot(rhs.m))

    def transpose(self) -> 'Matrix':
        '''Transpose a matrix'''
        nrows = len(self.m)
        ncols = len(self.m[0])
        T = [[0] * ncols for _ in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                T[j][i] = self.m[i][j]
        return Matrix(T)
    
    @classmethod
    def identity(self, n: int) -> 'Matrix':
        '''Create an n by n Identity matrix.'''
        I = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
        return Matrix(I)

    @classmethod
    def zeros(self, size: Sequence[int]) -> 'Matrix':
        '''Create and m by n matrix filled with zeros'''
        Z = [[0]*size[0] for _ in range(size[1])]
        return Matrix(Z)
    
    # 1.1
    def decomposition_LU(self) -> 'list[Matrix, Matrix, Matrix, int]':
        '''
        Perform LU decomposition for a square matrix.
        Returns a list of 3 matrices: [P, L, U, number_of_row_exhanges]
        '''
        U = Matrix(self.m.copy())
        P = Matrix.identity(len(U.m))
        number_of_row_exhanges = 0
        L = Matrix.identity(len(U.m))
        L = Matrix.zeros((len(U.m), len(U.m)))

        # step 1: choose a pivot.
        # take the kth column and find the element with the largest
        # absolute value of all the other elements in this column
        for i in range(len(U.m) - 1):
            max_row_idx = i
            max_elem = U.m[i][i]

            for cur_row in range(i, len(U.m)):
                cur_elem = U.m[cur_row][i]
                if abs(cur_elem) > abs(max_elem):
                    max_elem = cur_elem
                    max_row_idx = cur_row

            # step 2: bring the row with max |element| to the top and perform row swaps
            if max_row_idx != i:
                P.m[[max_row_idx, i]] = P.m[[i, max_row_idx]]
                U.m[[max_row_idx, i]] = U.m[[i, max_row_idx]]
                L.m[[max_row_idx, i]] = L.m[[i, max_row_idx]]
                number_of_row_exhanges += 1

            # step 3: perform row reduction
            for j in range(i + 1, len(U.m)):
                multiplier = U.m[j][i]/max_elem
                U.m[j] -= multiplier * U.m[i]
                L.m[j][i] = multiplier

        # fill the diagonal of L with 1's by adding an Identity matrix to it    
        L.m = L.m + Matrix.identity(len(L.m)).m

        return [P, L, U, number_of_row_exhanges]

    def solve_using_LU(self, P_: 'Matrix', L_: 'Matrix', U_: 'Matrix', b: Sequence) -> 'Matrix':
        P = P_.m.copy()
        L = L_.m.copy()
        U = U_.m.copy()
        
        b_hat = P.dot(b)
        # now solving LUx = b_hat, where Ux=z

        # first solve Ly = b_hat
        y = b_hat.copy()
        for i in range(0, len(L) - 1):
            for cur_row in range(i+1, len(L)):
                coeff = L[cur_row][i]/L[i][i]
                y[cur_row] -= y[i] * coeff

        # now solve Ux=y
        x = y.copy()

        for i in reversed(range(0, len(U))):
            for cur_row in range(0, i):
                coeff = U[cur_row][i]/U[i][i]
                x[cur_row] -= x[i] * coeff

        for i in range(len(x)):
            x[i] /= U[i][i]
        
        return Matrix(x)

    def inverse_matrix_using_LU(self) -> 'Matrix':
        '''Get the inverse of a matrix using LU decomposition'''
        P, L, U, _ = self.decomposition_LU()
        inverse = Matrix.zeros((len(self.m), len(self.m)))

        # Compute the inverse matrix column by column
        # LUx = b, where b is a column of an Identity matrix
        for i, b in enumerate(P.transpose().m):
            # Ld = b
            d = L.solve_using_LU(Matrix.identity(len(L.m)), L, Matrix.identity(len(L.m)), b)
            # Ux = d
            x = U.solve_using_LU(Matrix.identity(len(U.m)), Matrix.identity(len(U.m)), U, d.m)
            # x becomes an ith column of the inverse matrix
            inverse.m[i] = x.m

        return inverse.transpose()

    def det(self) -> float:
        '''Compute the determinant of a matrix using LU decomposition'''
        U, number_of_row_exhanges = self.decomposition_LU()[2:]
        result = 1.0

        # determinant = product of all the diagonal entries of U
        # It reverses sign in case we perform an odd number of permutations
        for i in range(len(U.m)):
            result *= U.m[i][i]
        if number_of_row_exhanges % 2 != 0:
            result *= -1

        return result

    # 1.2
    def tridiagonal_matrix_algorithm(self, b: Sequence) -> 'Matrix':
        '''Returns a solution vector x to Ax=b, where A is a tridiagonal matrix.'''
        b = b.copy()
        A = self.m.copy()
        x = Matrix.zeros((len(b), 1)).m[0]
        # Forward substitution
        for row_idx in range(1, len(self.m)):
            proportion = A[row_idx, row_idx - 1] / A[row_idx - 1, row_idx - 1]
            A[row_idx] -= A[row_idx - 1] * proportion
            b[row_idx] -= b[row_idx - 1] * proportion
        
        x[len(x) - 1] = b[len(x) - 1] / A[len(x) - 1, len(x) - 1]
        for i in reversed(range(len(x) - 1)):
            x[i] = (b[i] - A[i, i+1]*x[i+1]) / A[i, i]

        return Matrix(x)

    # 1.3
    def l1_norm(self) -> float:
        '''Return l1 norm of a matrix or a vector'''
        norm = 0
        if self.m.ndim == 2:
            for col_idx in range(len(self.m[0])):
                s = 0
                for row_idx in range(len(self.m)):
                    s += abs(self.m[row_idx, col_idx])
                if s > norm:
                    norm = s
            return norm
        ret = sum([abs(i) for i in self.m])
        return ret


    def simple_iteration_method(self, b: Sequence, eps=0.01) -> 'list[Matrix, int]':
        '''Solve a system of linear equations using the simple iteration method'''
        A = self.m.copy()

        # Create alpha and beta arrays
        beta = np.array([b[i]/A[i, i] for i in range(len(b))])
        alpha = []
        for i in range(len(A)):
            alpha.append([-A[i, j]/A[i, i] if i != j else 0 for j in range(len(A[0]))])
        alpha = Matrix(alpha)

        # Perform iterations
        x = beta.copy()
        iter_count = 0
        eps_k = 0
        
        use_norm = alpha.l1_norm() < 1
        while iter_count == 0 or eps_k > eps:
            iter_count += 1
            x_prev = x.copy()
            x = beta + alpha.m.dot(x_prev)

            if use_norm:
                eps_k = alpha.l1_norm() / (1 - alpha.l1_norm()) * Matrix(x - x_prev).l1_norm()
            else:
                eps_k = (x - x_prev).l1_norm()
        
        return x, iter_count

    def Seidel_method(self, b: Sequence, eps=0.01) -> 'list[Matrix, int]':
        '''Solve a system of linear equations using Seidel method'''
        A = self.m.copy()

        beta = np.array([b[i]/A[i, i] for i in range(len(b))])
        alpha = []
        for i in range(len(A)):
            alpha.append([-A[i, j]/A[i, i] if i != j else 0 for j in range(len(A[0]))])
        alpha = Matrix(alpha)

        c, d  = Matrix.zeros((len(A), len(A[0]))), Matrix.zeros((len(A), len(A[0])))
        for i in range(len(A)):
            for j in range(len(A[0])):
                if j >= i:
                    d.m[i, j] = alpha.m[i, j]
                else:
                    c.m[i, j] = alpha.m[i, j]
        
        inverse = Matrix(Matrix.identity(len(A)).m - c.m).inverse_matrix_using_LU()
        alpha = inverse.multiply(d)
        beta = inverse.multiply(Matrix(beta))
        x = beta.m.copy()
        iter_count = 0
        eps_k = 0
        use_norm = alpha.l1_norm() < 1

        while iter_count == 0 or eps_k > eps:
            iter_count += 1
            x_prev = x.copy()
            x = beta.m + alpha.m.dot(x_prev)

            if use_norm:
                eps_k = alpha.l1_norm() / (1 - alpha.l1_norm()) * Matrix(x - x_prev).l1_norm()
            else:
                eps_k = (x - x_prev).l1_norm()
        
        return x, iter_count

    # 1.4
    def Jacobi_rotation_method(self, eps=0.01) -> 'list[Matrix, Matrix, int]':
        '''Get eigenvectors and eigenvalues of a matrix'''
        iter_count = 0
        A = self.m.copy()
        n = len(A)

        u, u_k, a_k = Matrix.identity(n), Matrix.identity(n), Matrix(A)

        phi_k = 0
        max_elem_pos = [0, 0]
        max_elem = 0

        while math.sqrt(sum([a_k.m[i, j]**2 for i in range(n) for j in range(n) if i != j])) > eps:
            max_elem = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if i != j and abs(a_k.m[i, j]) > max_elem:
                        max_elem = abs(a_k.m[i, j])
                        max_elem_pos = [i, j]
        
            if a_k.m[max_elem_pos[0], max_elem_pos[0]] == a_k.m[max_elem_pos[1], max_elem_pos[1]]:
                phi_k = math.pi / 4
            else:
                phi_k = 0.5 * math.atan(2 * a_k.m[max_elem_pos[0], max_elem_pos[1]] / (a_k.m[max_elem_pos[0], max_elem_pos[0]] - a_k.m[max_elem_pos[1], max_elem_pos[1]]))
            
            u_k.m[max_elem_pos[0], max_elem_pos[0]] = math.cos(phi_k)
            u_k.m[max_elem_pos[1], max_elem_pos[1]] = math.cos(phi_k)
            u_k.m[max_elem_pos[0], max_elem_pos[1]] = math.sin(phi_k)
            u_k.m[max_elem_pos[1], max_elem_pos[0]] = -math.sin(phi_k)
            
            a_k.m = u_k.m @ a_k.m
            u_k = u_k.transpose()
            a_k.m = a_k.m @ u_k.m
            u.m = u.m @ u_k.m
            u_k = Matrix.identity(n)
            iter_count += 1
        
        return [a_k, u, iter_count]


if __name__ == '__main__':
    # 1.1
    a = []
    b = []
    with open('./test_1_1.test', 'r') as test_file:
        n = int(test_file.readline())
        for _ in range(n):
            vec = [int(num) for num in test_file.readline().split()]
            a.append(vec)
        b = [int(num) for num in test_file.readline().split()]
    
    A = Matrix(a)

    np.set_printoptions(suppress=True)

    P, L, U, n_swaps = A.decomposition_LU()
    print(f'Initial matrix A:\n{A}\n')
    print(f'b = {b}\n')
    print(f'Ax = b solution: {A.solve_using_LU(P, L, U, b)}\n')

    print(f'P:\n{P}\n')
    print(f'L:\n{L}\n')
    print(f'U:\n{U}\n')

    print(f'Inverse of A:\n{A.inverse_matrix_using_LU()}\n')
    print(f'Determinant of A: {A.det()}')


    # 1.2
    a = []
    b = []
    with open('./test_1_2.test', 'r') as test_file:
        n = int(test_file.readline())
        a.append([int(i) for i in test_file.readline().split()] + [0]*(n - 2))
        for i in range(0, n - 1):
            vec = [0] * i + [int(num) for num in test_file.readline().split()] + [0] * (n - 3 - i)
            a.append(vec)
        b = [int(num) for num in test_file.readline().split()]

    A = Matrix(a)
    np.set_printoptions(suppress=True)

    x = A.tridiagonal_matrix_algorithm(b)
    print(f'Solution to Ax=b: {x}')

    print(f'Ax = {A.multiply(x)}')
    print(f'b = {b}')


    # 1.3
    a = []
    b = []
    eps = 0.
    with open('./test_1_3.test', 'r') as test_file:
        n, eps = [float(i) for i in test_file.readline().split()]
        for _ in range(int(n)):
            vec = [int(num) for num in test_file.readline().split()]
            a.append(vec)
        b = [int(num) for num in test_file.readline().split()]
    
    A = Matrix(a)

    np.set_printoptions(suppress=True)

    result_iter = A.simple_iteration_method(b, eps)
    print('Simple iteration method:')
    print(f'x = {result_iter[0]} with epsilon = {eps} after {result_iter[1]} iterations\n')

    result_Seidel = A.Seidel_method(b, eps)
    print('Seidel method:')
    print(f'x = {result_Seidel[0]} with epsilon = {eps} after {result_Seidel[1]} iterations\n')


    # 1.4
    a = []
    b = []
    eps = 0.
    with open('test_1_4.test', 'r') as test_file:
        n, eps = [float(i) for i in test_file.readline().split()]
        for _ in range(int(n)):
            vec = [int(num) for num in test_file.readline().split()]
            a.append(vec)
        b = [int(num) for num in test_file.readline().split()]
    
    A = Matrix(a)
    np.set_printoptions(suppress=True)
    res = A.Jacobi_rotation_method(eps)
    print(f'Epsilon = {eps}')
    print(f'Number of iterations: {res[2]}', end='\n\n')
    print(f'Matrix of eigenvectors: \n{res[1].m}', end='\n\n')
    print(f'Matrix with eigenvalues on its diagonal:\n{res[0].m}', end='\n\n')
    
    eigenvalues = [res[0].m[i, i] for i in range(len(res[0].m))]
    print(f'Eigenvalues: {eigenvalues}', end='\n\n')

    first_column = np.array([res[1].m[i, 0] for i in range(len(A.m))])
    print(f'A * x_1 = {A.m @ first_column}')
    print(f'lambda_1 * x_1 = {eigenvalues[0] * first_column}', end='\n\n')

    second_column = np.array([res[1].m[i, 1] for i in range(len(A.m))])
    print(f'A * x_2 = {A.m @ second_column}')
    print(f'lambda_2 * x_2 = {eigenvalues[1] * second_column}', end='\n\n')

    third_column = np.array([res[1].m[i, 2] for i in range(len(A.m))])
    print(f'A * x_3 = {A.m @ third_column}')
    print(f'lambda_3 * x_3 = {eigenvalues[2] * third_column}', end='\n\n')

