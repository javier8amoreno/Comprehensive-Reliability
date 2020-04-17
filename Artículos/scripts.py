import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import math as mth




def histo_frec(data, nombre, show_table = True): 
    #Definimos variables globales, esta variable sera el dataframe que contendra la tabla de resultados del histrograma
    global matrix_frame_frec
    # Definimos la cantidad de elementos
    cant_elemt = data.count()
    # Definimos el numero de clases
    k = round(mth.sqrt(cant_elemt))
       
    # Definimos parametros basicos del histograma, tanto por puntos como por densidad de poblacion
    n, bins = np.histogram(data, bins=k)
    m, bins2 = np.histogram(data, bins=k, density=True)
    
    # Creamos variable para almacenar las clases
    clases=[]
    # Creamos variable para almacenar la tabla de resultados
    matrix =[]
    #Calculamos la probabilida de ocurrencia de los eventos
    prob =n/cant_elemt
    # Construimos el array de clases
    for i in range(0,k):
        par_val=(round(bins[i],4),round(bins[i+1],4))
        clases.append(par_val)
    # Construmios la tabla de datos almacenandola en la matrix
    for i in range(0,k):
        diccionario ={"Clases":clases[i],"# Puntos":n[i],"Probabilidad":round(prob[i],4), "Densidad":m[i]}
        matrix.append(diccionario)
    matrix_frame_frec = pd.DataFrame(matrix)
    
    #Construimos el histograma
    plt.hist(data, bins=k , density = True, rwidth=0.9)
    data.plot.kde(bw_method='silverman')
    plt.title("Histograma " + nombre)
    plt.xlabel('Clases')
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.75)
    
    if show_table==True:
        return plt.show(), matrix_frame_frec
    else:
        return plt.show()

def histo_cum(data, nombre, show_table=True):
    #Definimos variables globales, esta variable sera el dataframe que contendra la tabla de resultados del histrograma
    
    # Definimos la cantidad de elementos
    cant_elemt = data.count()
    # Definimos el numero de clases
    k = round(mth.sqrt(cant_elemt))
    # Definimos parametros basicos del histograma, tanto por puntos como por densidad de poblacion
    n, bins = np.histogram(data, bins=k)
    a =plt.hist(data, bins="auto" , density = True, cumulative= True, rwidth=0.9)
    cum =a[0]
    # Creamos variable para almacenar las clases
    clases=[]
    # Creamos variable para almacenar la tabla de resultados
    matrix =[]
    # Construimos el array de clases
    for i in range(0,k):
        par_val=(round(bins[i],4),round(bins[i+1],4))
        clases.append(par_val)
    for i in range(0,k):
        diccionario ={"Clases":clases[i],"# Puntos":n[i],"Probabilidad":cum[i]}
        matrix.append(diccionario)
    matrix_frame_cum = pd.DataFrame(matrix)
   
    #Construimos el histograma
    plt.title("Histograma acumulado " + nombre)
    plt.xlabel('Clases')
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.75)

    if show_table==True:
        return plt.show(), matrix_frame_cum
    else:
        return plt.show()

class Beta_Pert:
    '''
    In probability and statistics, the PERT distribution is a family of continuous 
    probability distributions defined by the minimum (a), most likely (b) and maximum (c) 
    values that a variable can take. It is a transformation of the four-parameter Beta distribution
    
    inputs:
    low - The minimum value(s) of the PERT.
    peak - The most-likely value(s) of the PERT.
    high - The maximum value(s) of the PERT.
    g - float (default 4.0) The PERT's gda parameter, smaller values give a wider probability spread.
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Beta Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta]
    alpha
    beta
    mean
    B - Beta function for PDF
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plts de survivel function
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    ''' 
    
    def __init__(self, low:None, peak:None, high:None, gamma=4.0):
            
            self.a = low
            self.b = peak
            self.c = high
            self.g = gamma
            self.range = (self.c -self.a)
            if self.a is None or self.b is None or self.c is None:
                raise ValueError('Parameters low, peak and high must be specified')
            if self.g <= 0:
                raise ValueError('g parameter should be greater than 0. By default is 4.0')

            self.mean = round((self.a + (self.g*self.b) + self.c) / (self.g + 2),4)

            if self.mean == self.b:
                self. alpha = self.beta = 3.0
            else:
                self.alpha = round(((self.mean-self.a)*(2*self.b-self.a-self.c))/((self.b - self.mean)*(self.c - self.a)),4)
                self.beta = round(self.alpha*(self.c - self.mean)/(self.mean - self.a),4)
            
            self.dist = ss.beta(self.alpha, self.beta, loc=self.a, scale=self.range)
            self.parameters = np.array([self.alpha, self.beta])
            self.median = round((self.a + ((2+self.g)*self.b) + self.c) / (4+self.g),4)
            self.mode = round(self.b,4)
            self.variance = round(((self.mean-self.a) * (self.c-self.mean)) / (self.g+4),4)
            self.skewness = round((2 * (self.beta - self.alpha) * np.sqrt(self.alpha + self.beta + 1)
            ) / ((self.alpha + self.beta + 2) * np.sqrt(self.alpha * self.beta)),4)
            
            self.kurt = ((self.g+2) * ((((self.alpha - self.beta)**2) * (self.alpha + self.beta + 1)
                ) + (self.alpha * self.beta * (self.alpha + self.beta + 2)))
            ) / (self.alpha * self.beta * (self.alpha + self.beta + 2) * (self.alpha + self.beta + 4))
            
            self.excess_kurtosis = round(6*((self.alpha-self.beta)**2*(self.alpha+self.beta +1
            )-(self.alpha*self.beta*(self.alpha+self.beta+2)))/(self.alpha*self.beta*(self.alpha+self.beta+2)*(self.alpha+self.beta+4)
            ) +4,4)          
            
            self.param_title = str('low='+str(self.a)+', peak='+ str(self.b)+', high='+str(self.c)+', Gamma='+ str(self.g))
            self.param_title_long = str('Beta Pert (low=' + str(self.a) + ', peak=' + str(self.b) + ', high='+str(self.c)+', Gamma='+ str(self.g)+')')

    def PDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating PDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        **plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        PDF - this is PDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        pdf = self.dist.pdf(xvals)
         
        if show_plot == False:
            return pdf
        else:
            plt.plot(xvals, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Beta PERT\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()
    
    def CDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating CDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        *plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        CDF - this is the CDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        cdf = self.dist.cdf(xvals)
        
        if show_plot == False:
            return cdf
        else:
            plt.plot(xvals, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative density')
            text_title = str('Beta PERT\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()

    def SF(self, xvals=None, show_plot=True, **kwargs):
            '''
            Plots the SF (survival function)

            Inputs:
            show_plot - True/False. Default is True
            xvals - value for calculating SF at x point
            *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 
            1000 elements will be created using these ranges. If nothing is specified then the range will be based on the 
            distribution's parameters.
            *plotting keywords are also accepted (eg. color, linestyle)
            Outputs:
            SF - this is the SF for x value
            The plot will be shown if show_plot is True (which it is by default).
            '''
            if xvals is None:
                xvals = np.linspace(self.a,self.c,1000)
            else:
                xvals = xvals
                show_plot = False

            sf = self.dist.sf(xvals)

            if show_plot == False:
                return sf
            else:
                plt.plot(xvals, sf, **kwargs)
                plt.xlabel('x values')
                plt.ylabel('Fraction surviving')
                text_title = str('Beta PERT\n' + ' Survival Function ' + '\n' + self.param_title)
                plt.title(text_title)
                plt.show()        
        
    def stats(self):
            """ 
            Returns the main statistical information from the PERT
            
            Inputs:
            None

            Outputs:
            Statistical information from the PERT
            """
            print('Descriptive statistics for Beta PERT distribution with low=', self.a, ', peak=', self.b,', high=', self.c)
            print('Mean = ', self.mean)
            print('Median =', self.median)
            print('Mode =', self.mode)
            print('Variance =', self.variance)
            print('Skewness =', self.skewness)
            print('Excess kurtosis =', self.excess_kurtosis)

    def random_samples(self, number_of_samples, random_state=None):
        """ Returns a randompy-sampled value from the PERT
        
        Parameters
        ----------
        number of samples: int (default 1)
            Indicates how many random values should be returned
        random_state: int (default none)
            Seed value for random sample RNG.
            
        Returns
        -------
        Array:
            Randomly sampled values from the PERT dristribution.
        """

        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        
        rvs = self.dist.rvs(size=number_of_samples, random_state=random_state)
        return rvs

class Triangular:
    '''
    In probability theory and statistics, the triangular distribution is a continuous 
    probability distribution with lower limit a, upper limit b and mode c, where a < b and a ≤ c ≤ b.
    
    inputs:
    low - The minimum value(s) of the PERT.
    peak - The most-likely value(s) of the PERT.
    high - The maximum value(s) of the PERT.
    g - float (default 4.0) The PERT's gda parameter, smaller values give a wider probability spread.
    param_title_long - Useful in plot titles, legends and in printing strings. eg. 'Beta Distribution (α=5,β=2)'
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta]
    alpha
    beta
    mean
    B - Beta function for PDF
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plts de survivel function
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    ''' 
    
    def __init__(self, low:None, peak:None, high:None):
            
            self.a = low
            self.b = peak
            self.c = high
            self.range = (self.c -self.a)
            
            if self.a is None or self.b is None or self.c is None:
                raise ValueError('Parameters low, peak and high must be specified')
            assert low<=peak<=high, 'Triangular "peak" must lie between "low" and "high"'

            self.dist= ss.triang((1.0*self.b - self.a)/(self.c - self.a), loc=self.a,scale=self.range)
            self.mean = self.dist.mean()
            self.median = self.dist.median()
            self.variance = self.dist.var()
            self.std = self.dist.std()
            self.param_title = str('low='+str(self.a)+', peak='+ str(self.b)+', high='+str(self.c))
            self.param_title_long = str('Triangular (low=' + str(self.a) + ', peak=' + str(self.b) + ', high='+str(self.c)+')')

    def PDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating PDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        **plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        PDF - this is PDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        pdf = self.dist.pdf(xvals)
         
        if show_plot == False:
            return pdf
        else:
            plt.plot(xvals, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str('Triangular\n' + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()
    
    def CDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating CDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        *plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        CDF - this is the CDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        cdf = self.dist.cdf(xvals)
        
        if show_plot == False:
            return cdf
        else:
            plt.plot(xvals, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative density')
            text_title = str('Triangular\n' + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()

    def SF(self, xvals=None, show_plot=True, **kwargs):
            '''
            Plots the SF (survival function)

            Inputs:
            show_plot - True/False. Default is True
            xvals - value for calculating SF at x point
            *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 
            1000 elements will be created using these ranges. If nothing is specified then the range will be based on the 
            distribution's parameters.
            *plotting keywords are also accepted (eg. color, linestyle)
            Outputs:
            SF - this is the SF for x value
            The plot will be shown if show_plot is True (which it is by default).
            '''
            if xvals is None:
                xvals = np.linspace(self.a,self.c,1000)
            else:
                xvals = xvals
                show_plot = False

            sf = self.dist.sf(xvals)

            if show_plot == False:
                return sf
            else:
                plt.plot(xvals, sf, **kwargs)
                plt.xlabel('x values')
                plt.ylabel('Fraction surviving')
                text_title = str('Triangular' + ' Survival Function ' + '\n' + self.param_title)
                plt.title(text_title)
                plt.show()        
        
    def stats(self):
            """ 
            Returns the main statistical information from the PERT
            
            Inputs:
            None

            Outputs:
            Statistical information from the PERT
            """
            print('Descriptive statistics for Beta PERT distribution with low=', self.a, ', peak=', self.b,', high=', self.c)
            print('Mean = ', self.mean)
            print('Median =', self.median)
            print('Variance =', self.variance)
            print('Standar Dev =', self.std)

    def random_samples(self, number_of_samples, random_state=None):
        """ Returns a randompy-sampled value from the PERT
        
        Parameters
        ----------
        number of samples: int (default 1)
            Indicates how many random values should be returned
        random_state: int (default none)
            Seed value for random sample RNG.
            
        Returns
        -------
        Array:
            Randomly sampled values from the PERT dristribution.
        """

        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        
        rvs = self.dist.rvs(size=number_of_samples, random_state=random_state)
        return rvs

class Uniform:
    '''
    In probability theory and statistics, the continuous uniform distribution or rectangular distribution is 
    a family of symmetric probability distributions. The distribution describes an experiment where there is 
    an arbitrary outcome that lies between certain bounds.
    
    inputs:
    low - The minimum value(s) of the PERT.
    high - The maximum value(s) of the PERT.
    param_title - Useful in plot titles, legends and in printing strings. eg. 'α=5,β=2'
    parameters - [alpha,beta]
    alpha
    beta
    mean
    B - Beta function for PDF
    variance
    standard_deviation
    skewness
    kurtosis
    excess_kurtosis
    median
    mode
    PDF() - plots the probability density function
    CDF() - plots the cumulative distribution function
    SF() - plts de survivel function
    stats() - prints all the descriptive statistics. Same as the statistics shown using .plot() but printed to console.
    random_samples() - draws random samples from the distribution to which it is applied. Same as rvs in scipy.stats.
    ''' 
    
    def __init__(self, low:None, high:None):
            
            self.a = low
            self.c = high
            self.range = (self.c -self.a)
            
            if self.a is None or self.c is None:
                raise ValueError('Parameters low, peak and high must be specified')
            assert low<high, 'The "low" value should be smaller than "high"'

            self.dist= ss.uniform(loc=self.a, scale=self.range)
            self.mean = self.dist.mean()
            self.median = self.dist.median()
            self.variance = self.dist.var()
            self.std = self.dist.std()
            self.param_title = str('low='+str(self.a)+', peak='+', high='+str(self.c))
            self.title_short = str('Uniform Distribution\n')

    def PDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the PDF (probability density function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating PDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        **plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        PDF - this is PDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        pdf = self.dist.pdf(xvals)
         
        if show_plot == False:
            return pdf
        else:
            plt.plot(xvals, pdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Probability density')
            text_title = str(self.title_short + ' Probability Density Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()
    
    def CDF(self, xvals=None, show_plot=True, **kwargs):
        '''
        Plots the CDF (cumulative distribution function)
        
        Inputs:
        show_plot - True/False. Default is True
        xvals - value for calculating CDF at x point
        *If xvals is specified, it will be used. If xvals is not specified then an array with 
        1000 elements will be created using these ranges. 
        *plotting keywords are also accepted (eg. color, linestyle)
        Outputs:
        CDF - this is the CDF for x value
        The plot will be shown if show_plot is True (which it is by default).
        '''
        if xvals is None:
            xvals = np.linspace(self.a,self.c,1000)
        else:
            xvals = xvals
            show_plot = False
        
        cdf = self.dist.cdf(xvals)
        
        if show_plot == False:
            return cdf
        else:
            plt.plot(xvals, cdf, **kwargs)
            plt.xlabel('x values')
            plt.ylabel('Cumulative density')
            text_title = str(self.title_short + ' Cumulative Distribution Function ' + '\n' + self.param_title)
            plt.title(text_title)
            plt.show()

    def SF(self, xvals=None, show_plot=True, **kwargs):
            '''
            Plots the SF (survival function)

            Inputs:
            show_plot - True/False. Default is True
            xvals - value for calculating SF at x point
            *If xvals is specified, it will be used. If xvals is not specified but xmin and xmax are specified then an array with 
            1000 elements will be created using these ranges. If nothing is specified then the range will be based on the 
            distribution's parameters.
            *plotting keywords are also accepted (eg. color, linestyle)
            Outputs:
            SF - this is the SF for x value
            The plot will be shown if show_plot is True (which it is by default).
            '''
            if xvals is None:
                xvals = np.linspace(self.a,self.c,1000)
            else:
                xvals = xvals
                show_plot = False

            sf = self.dist.sf(xvals)

            if show_plot == False:
                return sf
            else:
                plt.plot(xvals, sf, **kwargs)
                plt.xlabel('x values')
                plt.ylabel('Fraction surviving')
                text_title = str(self.title_short + ' Survival Function ' + '\n' + self.param_title)
                plt.title(text_title)
                plt.show()        
        
    def stats(self):
            """ 
            Returns the main statistical information from the PERT
            
            Inputs:
            None

            Outputs:
            Statistical information from the PERT
            """
            print('Descriptive statistics for ',self.title_short,' with low=', self.a,', high=', self.c)
            print('Mean = ', self.mean)
            print('Median =', self.median)
            print('Variance =', self.variance)
            print('Standar Dev =', self.std)

    def random_samples(self, number_of_samples, random_state=None):
        """ Returns a randompy-sampled value from the PERT
        
        Parameters
        ----------
        number of samples: int (default 1)
            Indicates how many random values should be returned
        random_state: int (default none)
            Seed value for random sample RNG.
            
        Returns
        -------
        Array:
            Randomly sampled values from the PERT dristribution.
        """

        if type(number_of_samples) != int or number_of_samples < 1:
            raise ValueError('number_of_samples must be an integer greater than 1')
        
        rvs = self.dist.rvs(size=number_of_samples, random_state=random_state)
        return rvs

def  met_sobre(val_x, val_y, df):

    slope, intercept, r_value, p_value, std_err = ss.linregress(val_x, val_y)
    line_kws="{0:.3f} * x + {1:.3f}".format(slope,intercept)


    val_up=0

    for i in np.arange(1,2,0.01):
        lim_sup = val_x.apply(lambda x: i*(slope* x + intercept))
        if pd.DataFrame.any(val_y >= lim_sup):
            val_up = round(i,2)
        else:
            continue
    line_kwsup="{2:.3f}*({0:.3f} * x + {1:.3f})".format(slope,intercept,val_up)


    val_low=0

    for i in np.arange(0,1,0.01):
        lim_low = val_x.apply(lambda x: i*(slope*x+intercept))
        if pd.DataFrame.any(val_y <= lim_low):
            continue
        else:
            val_low = round(i,2)

    line_kwslow="{2:.3f}*({0:.3f} * x + {1:.3f})".format(slope,intercept,val_low)
    
    formulas = line_kws,line_kwslow, line_kwsup
    plt.figure(figsize=(10,6))
    sns.regplot(x=val_x, y=val_y, data=df, fit_reg=True)
    plt.plot(val_x, [eval(formulas[1]) for x in val_x], color="r")
    plt.plot(val_x, [eval(formulas[2]) for x in val_x], color="r")
    plt.show()
    print(formulas)
    return formulas
