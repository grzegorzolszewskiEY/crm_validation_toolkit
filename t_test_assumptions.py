import statsmodels.api as sm
import numpy as np
import pandas as pd 
from scipy.stats import uniform, binom, norm, levene, bartlett, f, shapiro

#from t_test_assumptions import perform_test

#perform_test()

def perform_test(self, vec1, vec2, alpha):
    self.check_assumptions(vec1, vec2, alpha)
    pass

def check_assumptions(self, vec1, vec2, alpha):
    pass

class t_test_assumptions():

    """
    class check if t-test assumptions are met for two given vectors.
    Assumptions are:
    - normality - two compared populations following normal distribution 
    (under weak assumption this follows in large samples from the central limit theorem)
    normality can be checked for small sample size via shapiro wilk test and for large via Q-Q plot

    - homogenity of variances - both samples have approximately the same variance
    it is checked by performing F-test, Levene test, Bartlett test and Q-Q plot

    - independance
    """

    def __init__(self):
        pass

    def var_homogenity_qq(self, vec1, vec2):
        QQplot = sm.qqplot_2samples(data1 = vec1, data2 = vec2, line='45') 

    def var_homogenity(self, vec1, vec2, alpha):
        levene_test = levene(vec1, vec2)
        bartlett_test = bartlett(vec1, vec2)
        brown_testing = self.brown_forsythe_test(vec1, vec2)
        data = {'p-value': [levene_test[1], bartlett_test[1], brown_testing[1]]}
        test_df = pd.DataFrame(data=data, index = ['levene_test', 'bartlett_test', 'brown-forsythe_test'])
        #print(test_df)
        return ((test_df['p-value'] < alpha).sum() < 3)  #assumption that only one test need to pass[TRUE, TRUE, TRUE] -> 3 and we want to get <3 (one false) -> return True (1)
        
    def assumptions_met(self, vec1, vec2, alpha):
        var_homogenity_check = self.var_homogenity(vec1, vec2, alpha) #should be 1 (True) 
        #print('---VAR HOM---')
        #print(var_homogenity_check)
        normality_check = self.normality(vec1, vec2, alpha)  #should be 0 (False)
        if normality_check + var_homogenity_check == 1:
            return True
        else: 
            return False
  
    def brown_forsythe_test(self, vec1, vec2):
        """
        Performs the Brown-Forsythe test for the equality of variances between two samples x and y.
        
        Returns the test statistic and p-value.
        """
        # Calculate the absolute deviations from the median
        x_dev = abs(vec1 - np.median(vec1))
        y_dev = abs(vec2 - np.median(vec2))
        
        # Calculate the test statistic
        bf_stat, p_value = levene(x_dev, y_dev, center='median')
        
        return [bf_stat, p_value]
    
    def normality(self, vec1, vec2, alpha):
        #should it be even checked? see: Central Limit Theorem, n >= 30/50 (no compromise there)
        approve_list = []
        if len(vec1) <= 30 and len(vec1) > 3: #to perform shapiro wilk test you need sample size larger then 3
            if shapiro(vec1)[1] < alpha:
                approve_list.append(1)
            else:
                approve_list.append(0)

        if len(vec2) <= 30 and len(vec2) > 3: 
            if shapiro(vec2)[1] < alpha:
                approve_list.append(1)
            else:
                approve_list.append(0)

        if len(vec2) <= 3 or len(vec1) <= 3:
            approve_list.append(1)

        if len(vec1) > 30 and len(vec2) > 30:
            approve_list.append(0)

        else:
            approve_list.append(0)
            
        return sum(approve_list)
    

    

