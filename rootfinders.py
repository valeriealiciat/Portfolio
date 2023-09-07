import numpy as np
import matplotlib.pyplot as plt


# Q2 _____________________________________________________________________
def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Check certain that a and b have different signs
    # Check that Nmax is a positive integer
    if f(a)*f(b) >= 0 :
        raise ArithmeticError
    elif Nmax<=0:
        raise ValueError
        
    #Start the bisection method iteration
    #here, we use the 'for' loop
    n=0
    for n in range(1,Nmax+1):
        m = (a + b)/2
        fm = f(m)
        p_array[n-1]=m
        #we replace the elements in our zero array with iteration results
        
        #If f(m)>0, we replace b with m
        if f(a)*fm < 0:
            a = a
            b = m
            n+=1
        
        #If f(m)<0, we replace a with m
        elif f(b)*fm < 0:
            a = m
            b = b
            n+=1
        elif fm == 0:
            return p_array
    #We return the array with the sequence of approximations obtained 
    #by the fixed point iteration
    return p_array


# Q3 _______________________________________________________________________


def fixedpoint_iteration(f,c,p0,Nmax):

    
    """
    Fixed Point Iteration Method: Returns a numpy array of the 
    sequence of approximations obtained by the fixed point iteration method.
    
    Parameters
    ----------
    f : Function
        Input function which can be found in the given definition of 
        g(x)= p0 - c*f(x)
    c : Real number 
        Constant used in the given definition of g(x)= p0 - c*f(x) 
    p0: Real number 
        The initial value approximation to trigger the fixed point iteration.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """

    if c==0:
        raise ValueError
        
    elif Nmax<=0 or type(Nmax)!=int:
        raise ValueError
        
    #Start 
    #begin with an array of zeroes
    p_array = np.zeros(Nmax)
    n=0
    g = (lambda p0: p0 - c*f(p0))
    
    #for loop is used and will stop when n reaches Nmax+1
    for n in range(1,Nmax+1):
        if n <= Nmax:
            p1=g(p0)      #g(p0) becomes our new p1
            p0=p1         #p1 becomes our new p0
            p_array[n-1]=p1  #replace the elements in our zero array with iteration results
            n=+1
        
        elif n> Nmax:
            return p_array
        
    return p_array

# Q4 ____________________________________________________________________

def newton_method(f,dfdx,p0,Nmax):
    #Check if Nmax is a positive integer, otherwise we raise a ValueError
    if Nmax<=0 or type(Nmax)!=int:
        raise ValueError
        
    #Start 
    #we begin with an array of zeroes
    p_array = np.zeros(Nmax)
    n=0
    N=(lambda p0: p0 - f(p0)/dfdx(p0))      #N is our function of Newton Raphson Method
    
    #loop our function by using n and Nmax to break our loop
    for n in range(1,Nmax+1):

        p1=N(p0)
        p0=p1
        #change the elements in our zero array with iteration results
        p_array[n-1]=p1
        n=+1
    return p_array


# Q5 (with secant method below Q6) ________________________________________

# Q6 _______________________________________________________________________


def secant_method(f,p0,p1,Nmax):
    import math

    tol=1*math.exp(-14) #tolerance
    L=[p1] #we create an array with p1 as its first element
    
    #check if p0 and p1 is not the same value
    if p0==p1:
            raise ValueError

    elif f(p0)*f(p1) >= 0:  
            raise ArithmeticError
    elif Nmax<0:                   #Nmax should be a positive integer
            raise ValueError
            
    aa = p0
    bb = p1

    for n in range(1,Nmax):
        mm = bb - f(bb)*((bb - aa)/(f(bb) - f(aa)))   #we use the secant method formula

            
        if np.abs(f(bb)-f(aa))<tol: #to avoid division by zero, we first check if |f(bb)-f(aa)|<tol
            mm=bb
            L.append(bb)    #pn=pn-1 and we add that into our array
            
        
        elif n<Nmax:    #we stop the loop if n<Nmax
            aa = bb
            bb = mm
        
            L.append(bb)  

  
    return np.array(L)



# Q5 _____________________________________________________________________

def plot_convergence(p_exact,f,dfdx,c,p0,p1,Nmax,fig):
    
    #create an array of pexacts with Nmax elements
    pexact= [p_exact]* Nmax
    
    
    ne= newton_method(f,dfdx,p0,Nmax)
    fi= fixedpoint_iteration(f,c,p0,Nmax)
    bi= bisection(f,p0,p1,Nmax)
    se= secant_method(f,p0,p1,Nmax)
    
    #we calculate |pexact - pn| of each iteration
    new = (np.abs(pexact - ne))
    fix = (np.abs(pexact - fi))
    bis = (np.abs(pexact - bi))
    sec = (np.abs(pexact - se))

    #we create an array with chronological integers from 1 to Nmax
    #we plot the graphs using a scatter plot and assign different colors
    #and labels for each iteration methods and set labels for the two axises.
    
    xx= np.array(range(1,Nmax+1))
    ax = fig.add_subplot(111)
    ax.scatter(xx,new,color='r', label="Newton's Method")
    ax.scatter(xx,fix,color='b', label="Fixed Point Iteration Method")
    ax.scatter(xx,bis,color='y', label="Bisection Method")
    ax.scatter(xx,sec,color='m', label="Secant Method")
    ax.set_ylabel('|p-pn|')
    ax.set_xlabel('n')
    plt.xticks(range(1,Nmax+1))

    #we set the y axis to be logarithmic
    plt.semilogy()
    plt.legend(loc='upper right')



















