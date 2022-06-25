#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 02:01:04 2022

@author: valleriealiciat
"""
import numpy as np
import matplotlib.pyplot as plt

#%%

def backward_substitution(M,n):
    
    """
    Backward Substitution: Returns a numpy array of the 
    solution of a linear system (in echelon form).
    
    Parameters
    ----------
    M : numpy.ndarray
        Input matrix of size n x (n+1) (assumed to be in echelon form)
    n : integer
        Size of system
        
    Returns
    -------
    x : numpy.ndarray, shape (n,)
        Solution vector of augmented matrix M
    """
    
    # Initialise
    x=np.zeros([n,1])
    
    # First compute last one 
    x[n-1]=M[n-1,n]/M[n-1,n-1]
    
    # Loop backwards to compute others
    for i in range(n-2,-1,-1):
        x[i]=(M[i,n]-np.dot(M[i,i+1:n],x[i+1:n]))/M[i,i]
    
    # End
    return x

    
#%%
def no_pivoting(A,b,n,c):
    
    """
    Returns the augmented matrix M arrived at by starting from the augmented
    matrix [A b] and performing forward elimination without row interchanges
    until all of the entries below the main diagonal in the first c columns
    are 0.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the square matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        integer that is at least 2.
    c : integer
        positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        2-D array representing the matrix M.
    """
    
    # Create the initial augmented matrix
    M=np.hstack((A,b))
     
    #Here we calculate the forward part of the Gaussian elimination.
    for i in range(0, c):
        for j in range(i+1, n):
            
            #to find the multiplicatiom factor
            g = M[j,i] / M[i,i]
            
            for k in range(i, n+1):
                
            #row operation
                M[j,k] = M[j,k] - g * M[i,k]

    return M


def no_pivoting_solve(A,b,n):

    """
    No pivoting solve: Returns a numpy array of the solution of 
    a linear system after gaussian elimination without pivoting.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the square matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        integer that is at least 2.
        
    Returns
    -------
    x : numpy.ndarray, shape (n,1)
        Solution x of the linear system Ax=b computed using Gaussian
        elimination without pivoting.
    """
    M = no_pivoting(A, b,n,n)
    
    return backward_substitution(M, n) #solve the matrix using the given code in backward.py



#%% 
def find_max(M,n,i):
 
    p=0
    m=M[p,i] #here, we set m to be the first element in column i of the matrix
    for r in range(0,n):
 
        if np.abs(M[r,i])>m: #we compare the element in r-th row of the
                             #i-th column with m
                m=np.abs(M[r,i]) 
                p=r
                
                #set m to be the new element in the r-th row if it is greater than m
                #and set p to be the new row number
    return p


def partial_pivoting(A,b,n,c):
    '''
    Returns the augmented matrix M arrived at by starting from the augmented
    matrix [A b] and performing forward elimination using partial pivoting
    until all of the entries below the main diagonal in the first c columns
    are 0.

    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        array representing the square matrix A in the linear system Ax=b.
    b : numpy.ndarray of shape (n,1)
        array representing the vector b in the linear system Ax=b.
    n : integer
        integer that is at least 2.
    c : integer
        positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        2-D array representing the matrix M.
    '''
    
    M=np.hstack((A,b))
    
    #Steps are similar to no_pivoting
    
    for i in range(0, c):  #we use c instead to prematurely stop the function
        for j in range(i+1, n):
            
            p = find_max(M,n,i)  #use find_max to detect the row with absolute maximum
            M[[i,p],:] = M[[p,i],:]  #we switch the current row with the p-th row
            factor = M[j,i] / M[i,i]
            
            for k in range(i, n+1):
                M[j,k] = M[j,k] - factor * M[i,k] #row operation

    return M


def partial_pivoting_solve (A ,b , n ):
    
    M = partial_pivoting(A,b,n,n)
    return backward_substitution(M, n) #we use backward.py to solve the matrix after 
                    #Gaussian elimination with partial pivotiing



#%%

def Doolittle(A,n):
    
    L= np.identity(n) #we start with an identity matrix for L
    
    #The steps are similar to the code no_pivoting
    for i in range(0, len(A[0,:])): 
        for j in range(i+1, n):
            
            factor = A[j,i] / A[i,i]
            
            for k in range(i, len(A[0,:]+1)):
                #execute the row operations
                A[j,k] = A[j,k] - factor * A[i,k] #the result of A will be U (Upper triangular part)
                L[j,k] = L[j,k] + factor * L[i,k] #while L will be the Lower triangular part of the decomposition

    return L,A
#%%
def Gauss_Seidel(A,b,n,x0,tol,maxits):
    
    #lower triangular matrix from A
    L = np.tril(A) 
    
    #upper triangular matrix
    U = A - L  
    
    #to initialize while loop
    it=1
    
    #we stop when we reach maxits
    while it<=maxits: #we stop when we reach maxits
    
        #apply the formula for gauss seidel
        x0 = np.dot(np.linalg.inv(L), b - np.dot(U, x0)) 
        it+=1
        
        if np.max(np.abs(b-np.matmul(A,x0))) < tol:
            return x0
        
    if np.max(np.abs(b-np.matmul(A,x0))) > tol:
        x0=(" Desired tolerance not reached after maxits iterations have been performed.")
        return x0


#%% Test code

#%% No Pivoting Solve

# Initialise 
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
b = np.array([[7],[6],[4]])
n = 3

# Run no_pivoting
M = no_pivoting(A,b,n,1)
# Print output
print('\n')
print('No Pivoting 1 test output:')
print(M)

# Run no_pivoting
M = no_pivoting(A,b,n,2)
# Print output
print('\n')
print('No Pivoting 2 test output:')
print(M)

# Solve Ax=b
x = no_pivoting_solve(A,b,n)
# Print output
print('\n')
print('No Pivoting Solve test output:')
print(x)

# Help
help(no_pivoting_solve)


#%% Partial Pivoting 


# Initialise
A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])
b = np.array([[7],[6],[4]])
n = 3

# Run find_max
p = find_max(np.hstack((A,b)),n,0)
# Print output
print('\n')
print('Find Max 1 test output:')
print(p)

# Run partial_pivoting
M = partial_pivoting(A,b,n,1)
# Print output
print('\n')
print('Partial Pivoting 1 test output:')
print(M)

# Run find_max
p = find_max(M,n,1)
# Print output
print('\n')
print('Find Max 2 test output:')
print(p)

# Run partial_pivoting
M = partial_pivoting(A,b,n,2)
# Print output
print('\n')
print('Partial Pivoting 2 test output:')
print(M)

# Solve Ax=b
x = partial_pivoting_solve(A,b,n)
# Print output
print('\n')
print('Partial Pivoting Solve test output:')
print(x)


#%% Doolittle


# Initialise
A = np.array([[1,1,0.0],[2,1,-1],[0,-1,-1]])
n = 3

# Run Doolittle
L, U = Doolittle(A,n)
# Print output
print('\n')
print('L:')
print(L)
print('\n')
print('U:')
print(U)


#%% Gauss Seidel


# Initialise
A = np.array([[4,-1,0],[-1,8,-1],[0,-1,4]])
b = np.array([[48],[12],[24]])
n = 3
x0 = np.array([[1.0],[1],[1]])
tol=1e-2


# Run Gauss_Seidel
x = Gauss_Seidel(A,b,n,x0,tol,3)
# Print output
print('\n')
print('Gauss Seidel 1 test output:')
print(x)

# Run Gauss_Seidel
x = Gauss_Seidel(A,b,n,x0,tol,4)
# Print output
print('\n')
print('Gauss Seidel 2 test output:')
print(x)


