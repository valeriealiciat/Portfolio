"""
MATH2033 Coursework 4 polynomial_interpolation module
"""

### Load other modules ###
import numpy as np
import matplotlib.pyplot as plt
### No further imports should be required ###

### Comment out incomplete functions before testing ###

#%%
def richardson( f, x0, h, k ):
    """
    Richardson: Returns a float, which is the approximation Nk(h) to f'(x0) using the 
    method of Richardson Approximation for levels k>0.

    Parameters
    ----------
    f : function
        function which its derivative is to be approximated.
    x0 : float
        point of evaluation.
    h : float
        the step size.
    k : integer
        the level of approximation.

    Returns
    -------
    deriv_approx : float
        the approximation Nk(h) to f'(x0).

    """
    #calculate for N1(h)
    d = lambda y: ( f( x0+y ) - f( x0-y ) ) / (2 * y)
    
    if k==1:
        deriv_approx= d(h)
    
    #calculate for Nk(h) k>1
    elif k>1:
        
        #define the required given formulas
        a = lambda k: 1 / ( 1 - 4 ** ( k-1 ))
        b = lambda k: -( (4 ** ( k-1 )) / ( 1 - 4 ** (k-1)))
        
        #initiate loop
        for k in range ( 2 , k+1 ):
    
        
            if k==2:

                #formula for N1(h) is used to approximate N2(h) 
                deriv_approx= (a(k) * d(h)) + (b(k) * d(h/2)) 
                
                
            else:
                #Nk formula used to find deriv_approx
                Nk= lambda k,h: (a(k) * d(h)) + (b(k) * d(h/2)) 
                
                #formula to calculate Nk(h) is used
                deriv_approx= (a(k) * Nk(k-1,h)) + (b(k)*Nk(k-1, h/2)) 
         
    #return results        
    return deriv_approx

#%%
def richardson_errors(f,f_deriv,x0,n,h_vals,k_max):
    '''
    
    a.) the results shows that the errors calculated for the values in a.) increases in general and will reach a plateau.
    While k=1 and k=4 starts not far from each other at about h=10**-8, k=2 and k=3 starts from a relatively low error
    compared to k=1 and k=4. k=2 then begins to have a steep increase at around h=10^-4 and k=3 at around h=10^2.5 
    After reaching between h=10^-1 and h=10^-0, we can see how the error for levels 1,2,3 and 4 converges towards 
    the same value and thus remains stagnant. 
    
    b.) According to the figure, there is a positive correlation between h and error for all the levels 1,2,3 and 4.
    The figure shows that the level with the highest error being k=1, then followed by k=2, and followed closely
    by k=4 and k=3.
    


    '''
    #create an array for N1(hi)
    E1 = [ abs( f_deriv(x0) -  ( f( x0+h ) - f( x0-h ) ) / (2 * h) ) for h in h_vals]
    
    #create an array for Nk(hi), k>1
    E2 = [ abs( f_deriv(x0) - richardson(f, x0, h, k))  for k in range( 2 , k_max+1 ) for h in h_vals ]
    
    #combine E1 and E2
    E = E1 + E2
    
    #reshape the combined matrix into a shape of (k_max,n)
    error_matrix= np.reshape(E,(k_max,n))    
    
    fig = plt.figure() #This line is required (once) before any other plotting commands
    
    #set the title, and labels
    plt.title("Richardson Error")
    
    plt.xlabel('h')
    
    plt.ylabel('Error')
    
    #initiate loop to plot error with plt.loglog
    for k in range(0,k_max):
        
        K = k + 1
        
        plt.loglog(h_vals , error_matrix[k,:], label= 'error for k = %d' %K)
        
    plt.legend()
    
        
    #return error_matrix and figure
    return error_matrix, fig

#### Your submission should have no code after this point ####



