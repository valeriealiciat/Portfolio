#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 01:32:27 2022

@author: valleriealiciat
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
def lagrange_poly(p,xhat,n,x,tol):
        
    """
    Lagrange Polynomial: Returns a matrix with elements obtained 
    from the evaluated Lagrange polynomial formula and an error flag.
    
    Parameters
    ----------
    p : integer
        Number of distinct nodal points.
        
    xhat : numpy.ndarray
        Array containing the nodal points.
           
    n : integer
        Number of evaluation points.
        
    x : numpy.ndarray
        Array containing the evaluation points.
        
    tol : float
        Error tolerance.
        
    Returns
    -------
    lagrange_matrix : numpy.ndarray of shape (p+1,n)
                 A matrix where the ij-th entry is Li(xj) of the 
                calculated Lagrange polynomial.
    error_flag : integer
                 returns 0 if the nodal points are distinct and otherwise 1.
                
    """
    
    #initialize by creating an empty array of shape (p+1,n) to contain our results

    lagrange_matrix = np.empty( ( p+1, n ) , dtype = np.float64 )
    error_flag = 0
    error = 1
    
    
    #Check for errors
    for o in range(p):
        
        if abs( xhat[o] - xhat[o+1] ) < tol:
            
            error_flag = error


    #Start loops
    for t in range(n):
        
        for r in range(p+1) :  
            
            #Lagrange value reseted to 1 every time r loops
            lagrange = 1.0
            
            #Calculate Lagrange polynomial
            for g in range( p+1 ):
                
                if g != r :
                    
                    #Lagrange polynomial formula
                    lagrange = lagrange * (x[t] - xhat[g]) / (xhat[r] - xhat[g])
            
            #Store results
            lagrange_matrix[r, t] = lagrange

    return lagrange_matrix, error_flag


#%%
def uniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    
    """
    Uniform Polynomial Interpolation: Returns an array with elements obtained 
    from the evaluated pth order polynomial interpolant of a function at a set
    of points with a uniformly spaced nodal interpolation points, and an a 
    figure if produce_fig is True.
    
    Parameters
    ----------
    a : float
        initial value of the nodal interpolation interval.
        
    b : float
        final value of the nodal interpolation interval.
        
    p : integer
        order of polynomial interpolant.
        
    n : integer
        number of evaluation points.
        
    x : numpy.ndarray
        a set of evaluation points.
        
    f : function
        function in the polynomial interpolation formula.
        
    produce_fig : Boolean
        logical truth values. (False or True)
        
        
    Returns
    -------
    interpolant : numpy.ndarray of shape (n, )
                returns an array of interpolant points obtained from the
                polynomial interpolation.
                
    fig : matplotlib.figure.Figure
            returns a graph of interpolation results when the Boolean 
            input produce_fig is True and otherwise None.
                 
                
    """
    
    #Create an empty list to store results and xhat is defined.
    L = []
    xhat=np.linspace( a , b , p+1)
    
    #Store results from lagrange_poly
    lp = lagrange_poly(p,xhat,n,x,1e-10)[0]
    
    
    #Start loops
    for o in range( n ):
             
            #po value restarted to 0 every o loop
            po = 0
            
            #Calculate Lolant
            for g in range( p+1 ):
                
                #Use interpolant formula
                po += f( xhat[g] ) * lp[ g,o]
                
            #Store results                
            L.append(po) 

    interpolant = np.array(L) 
    
    
    #Return interpolant results and plot figure if produce_fig is True
    if produce_fig==True: 
        
        #Initialize plot
        fig = plt.figure()
        
        #plot, set the title, labels and legend
        plt.title('Uniform Polynomial Interpolation') 
        
        plt.xlabel('x')
        
        plt.ylabel('y')
        
        plt.plot(x, f(x), "o",  label='f(x)')
        
        plt.plot(x, f(x), label='Pp(x)')
        
        plt.legend()
        
        return interpolant, fig 
    
    
    # Return interpolant results and 'None' if produce_fig is False
    elif produce_fig==False: 
        
        return interpolant, None 

#%%
def nonuniform_poly_interpolation(a,b,p,n,x,f,produce_fig):
    
    """
    Nonuniform Polynomial Interpolation: Returns an array with elements obtained 
    from the evaluated pth order polynomial interpolant of a function at a 
    set of points with nonuniformly spaced nodal interpolation points and
    and an a figure if produce_fig is True.
    Parameters
    ----------
    a : float
        initial value of the nodal interpolation interval.
        
    b : float
        final value of the nodal interpolation interval.
        
    p : integer
        order of polynomial interpolant.
        
    n : integer
        number of evaluation points.
    
    x : np.ndarray
        a set of evaluation points.
        
    f : function
        function in the polynomial interpolation formula.
        
    produce_fig : Boolean
        logical truth values. (True or False)

    Returns
    -------
    nu_interpolant : numpy.ndarray of shape (n,)
        returns a set of interpolant points obtained from the
        polynomial interpolation formula.
    
    fig : Figure
        returns a graph of interpolation results when the Boolean 
        input produce_fig is True and otherwise None.

    """
    
    #Create empty lists to contain results of evaluation.
    L=[]
    
    #Use given xhat formula.
    xh = [ np.cos( ( (2*k+1) / (2*(p+1)) ) * np.pi ) for k in range (0,p+1) ]
    
    #Calculate mapped xhat
    xhat = [ ( (b-a) / 2 ) * x + ( (a+b) / 2 ) for x in xh]
    
    #Store lagrange polynomial results
    lp = lagrange_poly(p, xhat, n, x, 1e-10)[0]
    
    #Evaluate F by values in xhat
    F = [ f(x) for x in xhat ]


    #Start loops
    for j in range( n ): 
        
        #po value is resetted to 0 every j loop
        po=0 
        
        
        #Calculate interpolant
        for i in range( p+1 ):
            
            #Use interpolation formula
            po += F[ i ] * lp[ i,j ]
            
        #Store results    
        L.append(po) 

    nu_interpolant = np.array(L)
    
       
    #Return interpolant results and figure if produce_fig is True
    if produce_fig==True: 
        
        #Initialize plot
        fig = plt.figure()
        
        #plot, set plot title, labels, and legend
        plt.title('Non Uniform Polynomial Interpolation') 
        
        plt.xlabel('x')
        
        plt.ylabel('y')
        
        plt.plot(x, f(x), "o" , label = 'f(x)')
        
        plt.plot(x,nu_interpolant, label='Pp(x)')
        
        plt.legend()
        
        return nu_interpolant, fig
    
    #Return interpolant results and 'None if produce_fig is False
    elif produce_fig==False: 
        
        return nu_interpolant, None
    
  

#%%
def compute_errors(a,b,n,P,f):
    '''
    Compute the error |p_p(x) - f(x)| for both uniform and non uniform polynomial interpolation.

    Parameters
    ----------
    a : float
        initial value of the nodal interpolation interval.
        
    b : float
        final value of the nodal interpolation interval.
        
    n : integer
        number of evaluation points.
        
    P : np.ndarray
        polynomial degree.
        
    f : function
        function in the polynomial interpolation formula.

    Returns
    -------
    error_matrix : np.ndarray
        a matrix containing computed errors of shape (2,n), where the first row and second row
        contains the errors computed for a uniform and non uniform set of interpolating nodes
        respectively.
        
    fig : Figure
        returns a graph of computed errors based on the error matrix when the Boolean 
        input produce_fig is True and otherwise None.

    '''
    
    #Compute evaluation points
    data = np.linspace(a,b,2000)
    
    
    #Calculate f(x)
    mat = [f(d) for d in data]
    
    
    #Calculate error for points in 'data' for both uniform and non-uniform polynomial interpolation
    eu = [ max ( np.abs( mat - 
                        uniform_poly_interpolation(a,b,P[i],2000,data,f,False)[0] ) ) 
                              for i in range(n) ]
    enu = [ max( np.abs( mat - 
                        nonuniform_poly_interpolation(a,b,P[i],2000,data,f,False)[0] ) ) 
                               for i in range(n) ]
    
    #Combine arrays containing calculated errors
    error_matrix= np.vstack((eu,enu))
    
    
    #Initialize plotting
    fig = plt.figure()
    
    #set plot title, labels, and legend
    plt.title('Lagrange Interpolation Errors') 
    
    plt.xlabel('p')
    
    plt.ylabel('error')
    
    #plot using plt.semilogy
    plt.semilogy(P , error_matrix[0,:], label='Uniform Polynomial Interpolation')
    
    plt.semilogy(P , error_matrix[1,:], label='Non Uniform Polynomial Interpolation')
    
    plt.legend()

    return error_matrix, fig
    

#%%
def piecewise_interpolation(a,b,p,m,n,x,f,produce_fig):
    
    """
    Piecewise Interpolation: Returns an array with elements obtained 
    from the evaluated pth order  continuous piecewise interpolant of a function
    with uniformly spaced subintervals and an a figure if produce_fig is True.
    Parameters
    ----------
    a : float
        initial value of the nodal interpolation interval.
        
    b : float
        final value of the nodal interpolation interval.
        
    p : integer
        order of lagrange interpolant.
    
    m : integer
        number of subintervals
        
    n : integer
        number of evaluation points.
    
    x : np.ndarray
        a set of evaluation points.
        
    f : function
        function in the polynomial interpolation formula.
        
    produce_fig : Boolean
        logical truth values. (True or False)

    Returns
    -------
    nu_interpolant : numpy.ndarray of shape (n,)
        returns a set of interpolant points obtained from the
        piecewise interpolation
    
    fig : Figure
        returns a graph of interpolation results when the Boolean 
        input produce_fig is True and otherwise None.

    """
    #Calculate subintervals

    sub= np.linspace(a,b,m+1)

    #Initialize with zero matrix to increase speed of the code
    p_interpolant = np.zeros(n)

    for k in range(0,m):
        if k==0: #treat first interval differently
            indices = (x >= sub[k]) & (x <= sub[k+1])
        else:
            indices = (x > sub[k]) & (x <= sub[k+1])

        #Find the correct nodal points to ensure the function is continuous
        widthdiff = (sub[k+1]-sub[k])*(1.0/np.cos(np.pi/(2*(p+1)))-1)
        leftnode = sub[k]-widthdiff/2.0
        rightnode = sub[k+1]+widthdiff/2.0

        #Build interpolant over the entire range of points in x
        interp, unused_fig = nonuniform_poly_interpolation(leftnode,rightnode,p,n,x,f,False)

        #compute the interpolant
        p_interpolant += indices*interp

    
    #Return interpolant results and figure if produce_fig is True
    if produce_fig==True: 
        
        #Initialize plot
        fig = plt.figure()
        
        #plot, set plot title, labels, and legend
        plt.title('Piecewise Polynomial Interpolation') 
        
        plt.xlabel('x')
        
        plt.ylabel('y')
        
        plt.plot(x , f(x), "o" , label = 'f(x)')
        
        plt.plot(x , p_interpolant, label='Pp(x)')
        
        plt.legend()
        
        return p_interpolant, fig
    
    #Return interpolant results and 'None if produce_fig is False
    elif produce_fig==False: 
        
        return p_interpolant, None



## TEST CODE

#%% Lagrange Polynomial
# Initialise
p = 3
xhat = np.linspace(0.5,1.5,p+1)
n = 7
x = np.linspace(0,2,n)
tol = 1.0e-10
#Run the function
lagrange_matrix, error_flag = lagrange_poly(p,xhat,n,x,tol)

print('Lagrange Polynomial test output:')
print(lagrange_matrix)
print(error_flag)

print(type(tol))
#%% Uniform Polynomial Interpolation
# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
interpolant, fig = uniform_poly_interpolation(a,b,p,n,x,f,True)

print('\n')
print('Uniform Polynomial Interpolation test output:')
print(interpolant)

#%% Non Uniform Polynomial Interpolation
# Initialise
a = 0.5
b = 1.5
p = 2
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
nu_interpolant, fig = nonuniform_poly_interpolation(a,b,p,n,x,f,True)

print('\n')
print('Non Uniform Polynomial Interpolation test output:')
print(nu_interpolant)

#%% Errors
# Initialise
n = 5
P = np.arange(1,n+1)
a = -1
b = 1
f = lambda x: 1/(x+2)
#Run the function
error_matrix, fig = compute_errors(a,b,n,P,f)
print('\n')
print('Errors test output:')
print(error_matrix)

#%% Piecewise Interpolation
#Initialise
p = 10
a = -1
b = 1
n = 7
m = 5
x = np.linspace(-0.9,0.9,7)
f = lambda x: 1/(x+2)
#Run the function
p_interpolant, fig = piecewise_interpolation(a,b,p,m,n,x,f,True)
print('\n')
print('Piecewise Interpolation test output:')
print(p_interpolant)




