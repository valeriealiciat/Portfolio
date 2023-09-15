import numpy as np
import matplotlib.pyplot as plt
import newton #(run newton_solver(...) as newton.newton_solver(...))

def theta_ode_solver(a,b,f,df,N,y0,theta):
    '''
    Theta Ode Solver : Returns an array of the approximation for the solution y of the
    initial value problem using the given initial condition by appylying theta-schemes.

    Parameters
    ----------
    a : integer
        initial point of the interval of evaluation.
    b : integer
        final point of the interval of evaluation.
    f : function
        function to solve the initial value problem.
    df : function
        derivative to the function stored in f.
    N : integer
        number of time-steps to take.
    y0 : float
        initial condition.
    theta : float
        variable in the closed interval of [0,1]
        

    Returns
    -------
    t : numpy.ndarray
        an array of shape (N+1,) containing evaluation values with time-step (b-a)/N.
    y : numpy.ndarray
        an array of shape (N+1,) containing approximation results where the ith element is the 
        approximation of the solution at ti, y(ti).

    '''
    #calculate the step size
    h = (b-a) / N
    
    #create an array for t in the interval [a,b] with step size h and size (N+1,)
    t = np.array([ a + h*i for i in range(N+1)])
    
    #create an array of zeroes to store approximations
    y = np.zeros(N+1)
    
    #set the first element to be the initial condition, y0
    y[0] = y0
    
    #initialize loop to approximate solutions
    for i in range(N):
        
        #Forward Euler's formula used for initial guess in newton's method
        yf = y[i] + h * theta * f(t[i],y[i]) 
        
        #alter the given theta-scheme formula so that it equals 0, F(yii)=0
        F = lambda yii: yii - y[i] - ( h * theta * f( t[i],y[i]) ) - ( h * (1-theta) * f( t[i+1], yii ) )
        
        #find the derivative of F(yii), dF
        dF = lambda yii: 1 - ( h * theta * df( t[i],y[i]) ) - ( h * (1-theta) * df( t[i+1], yii ) )
    
        #use newton's method to solve F(yii)=0 and update the elements with the result.
        y[i+1]=newton.newton_solver(F,dF,yf,100,1e-12)
        
    return t, y
    
    

def runge_kutta(a,b,f,N,y0,m,method):   
    """
    Runge Kutta: Using the Runge Kutta method, this functions returns a matrix containing
    the approximate solutions to the initial value problem with the given initial 
    conditions depending on the method chosen.

    Parameters
    ----------
    a : integer
        initial point of the interval of evaluation.
    b : integer
        final point of the interval of evaluation.
    f : numpy.ndarray
        array containing functions to solve the initial value problem.
    N : integer
        number of time-steps to take.
    y0 : numpy.ndarray
        array containing initial conditions.
    m : integer
        the number of functions in f and initial conditions.
    method : integer
        The integers indicates the method used to solve the initial value problem
        the numbers used means the following:
        - 1 indicates the usage of Forward Euler Method,
        - 2 indicates the usage of Midpoint Method, and
        - 3 indicates the usage of Heun's 3rd order method.

    Returns
    -------
    t : numpy.ndarray
        an array of shape (N+1,) containing evaluation values with time-step (b-a)/N.
    y : numpy.ndarray
        a matrix array y[i,j] of shape (m,N+1) where the i-1th row of the 
        matrix contains the approximations to the ith component of y at all time points.

    """
    #calculate the step size.
    h = (b-a) / N
    
    #create an array for t in the interval [a,b] with step size h and size (N+1,).
    t = np.array([ a + h*i for i in range(N+1)])
    
    #create a matrix array of zeroes of shape (N+1,m) to store approximations.
    Y = np.zeros((N+1,m))
    
    #set the first row of Y to be y0.
    Y[0,:]= y0
    
    #Forward Euler's method

    #initialize iteration which is done per row
    for i in range(N):
        
        if method==1:
            #Forward Euler's formula
            Y[i+1,:] = Y[i,:] + h * f(t[i],Y[i,:])
        
        elif method==2:
            #Midpoint Method formula
            Y[i+1,:] = Y[i,:] + h*f(t[i] +h/2 , Y[i,:] +h/2*f(t[i],Y[i,:]))
        
        elif method==3:
            #Heun's 3rd order formula
            Y[i+1,:] = Y[i,:] + h/4 * ( f(t[i],Y[i,:]) + 3*f( t[i] + 2*h/3 , Y[i,:] + 2*h/3 * f(t[i] + h/3, Y[i,:] + h/3 *f( t[i],Y[i,:] )   ))) 
        
    #transpose the matrix
    y= np.transpose(Y)
    
    return t, y


