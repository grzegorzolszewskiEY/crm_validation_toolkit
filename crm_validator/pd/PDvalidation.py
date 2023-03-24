import numpy as np
import pandas as pd 
from scipy.stats import uniform, binom, norm, beta
from crm_validator.util.gAUC_class import gAUC
#import LGDvalidator
import matplotlib.pyplot as plt
#from LGD_validator import LGD_Validator
from sklearn import metrics
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")
import random
from textwrap import wrap




class PD_validator(object):
    def __init__(self, data:pd.DataFrame):
        self.data = data[data.filter(regex='^(?!Unnamed)').columns]
        self.data = self.cleaned_data(data)
        self.defaulted_customers = np.array(self.data.loc[self.data['Default_status'] == 1, 'Default_status'])
        self.ratings_defaulted = np.array(self.data.loc[self.data['Default_status'] == 1, 'Previous_rating'])
        self.nondefaulted_customers = np.array(self.data.loc[self.data['Default_status'] == 0, 'Default_status'])
        self.ratings_nondefaulted = np.array(self.data.loc[self.data['Default_status'] == 0, 'Previous_rating'])
        self.def_or_not = self.data['Default_status'].notnull()
        self.good_indices = self.def_or_not[self.def_or_not].index.values
        self.def_or_not = np.array(self.data.loc[self.good_indices, 'Default_status'])
        self.PD_score = np.array(self.data.loc[self.good_indices, 'PD'])
        
        self.prev_ratings = list(self.data.Previous_rating.unique())
        self.ratings = self.prev_ratings + [len(self.prev_ratings) + 1, 'Different', 'Terminated']
        
        
        self.row = [[0 for i in range(len(self.ratings))] for j in range(len(self.prev_ratings))]
        
        for i in range(len(self.prev_ratings)):
            self.row[i] = [len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(1)]), len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(2)]), 
           len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(3)]), len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(4)]),
           len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(5)]), len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(6)]),
           len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(7)]), len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq(8)]), len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq('Different')]),
           len(self.data.loc[self.data.Previous_rating.eq(i+1) & self.data.Rating.eq('Terminated')])]
              
        self.exposure_per_grade = np.zeros(len(self.prev_ratings))
        for i in range(1, len(self.prev_ratings) + 1):
            self.exposure_per_grade[i-1] = self.data.loc[self.data['Previous_rating'] == i, 'original_exposure'].sum()
        
        self.N_per_grade = np.array([sum(self.row[0]), sum(self.row[1]), sum(self.row[2]), sum(self.row[3]), sum(self.row[4]), sum(self.row[5]), sum(self.row[6])])       

    def qualitative_validation1(self, data):
        M = len(data)
        M_ex = len(data.loc[data['process_deficiency'] == 1, 'process_deficiency'])
        N = M - M_ex
        return(['The size of the portfolio (before data exclusions) is ', M,
                'The number of customers which are excluded from the validation sample due to a rating process deficiency is ', M_ex,
                'The number of customers in the validation sample after data exclusions is therefore ', N])
        
    def qualitative_validation2(self, data):
        M_def = len(data.loc[data['technical_defaults'] == 1, 'technical_defaults'])
        return(['The amount of customers with technical defaults is ', M_def])
        
    def cleaned_data(self, data):
        data = data.drop(data[data['technical_defaults'] == 1].index)
        data = data.drop(data[data['process_deficiency'] == 1].index) 
        return data
    
    def p_value(self, N: int, D:int , PD: float):
        a = D + 1/2
        b = N - D + 1/2
        p = beta.cdf(PD, a, b)
        return p
    
    def jeffreys_test(self, defaults_or_not: np.array, array_w_PD: np.array):
        D = sum(defaults_or_not)
        N = len(defaults_or_not)
        PD = array_w_PD.mean()
        p = self.p_value(N, D, PD)
        return [p, N, D]
    
    def summary_pred_ability(self, alpha: float):
        rows = [1, 2, 3, 4, 5, 6, 7, 'Portfolio']
        init_list = [0 for i in range(len(rows))]
        column_names = {'p-value': init_list, '# of customers' : init_list, '# of defaults': init_list}
        summary_table = pd.DataFrame(column_names, index = rows)
        for i in range(len(summary_table)):
            if i < 7:
                defaults_or_not = self.data.loc[self.data['Previous_rating'] == i+1, 'Default_status']
                indices = defaults_or_not.index.values
                defaults_or_not = defaults_or_not[~np.isnan(np.array(defaults_or_not))]
                defaults_or_not = np.array(defaults_or_not)
                PD = np.array(self.data.loc[indices, 'PD'])
                summary_table.iloc[i, :] = self.jeffreys_test(defaults_or_not, PD)
            else:
                defaults_or_not = np.array(self.data['Default_status'])
                defaults_or_not = defaults_or_not[~np.isnan(defaults_or_not)]
                PD = np.array(self.data['PD'])
                PD = PD[~np.isnan(PD)]
                summary_table.iloc[i, :] = self.jeffreys_test(defaults_or_not, PD)
                
        summary_table['is H0 rejected for alpha= {}'.format(alpha)] = summary_table['p-value'] < alpha 
        #fig = plt.figure(figsize = (10, 0.3))
        #ax = fig.add_subplot(111)
        #ax.table(cellText = summary_table.values, rowLabels = summary_table.index, colLabels = summary_table.columns, cellLoc = 'center')
        #ax.set_title("\n".join(wrap("Jeffreys test both at individual grade level and at portfolio level. The null hypothesis is that the estimated PD is greater than the true one", 60)))
        #ax.axis('off')
        return summary_table
        #return fig
    
    
    def mannwhitney_auc(self):
        statistic = 0
        count = 0
        for i in range(len(self.defaulted_customers)):
            for j in range(len(self.nondefaulted_customers)):
                if self.ratings_defaulted[i] < self.ratings_nondefaulted[j]:
                    count = count + 1
                elif self.ratings_defaulted[i] == self.ratings_nondefaulted[j]:
                    count = count + 1/2
                else:
                    count = count + 0
        statistic = statistic + count
    
        return (statistic/(len(self.defaulted_customers) * len(self.nondefaulted_customers)))
    
    def mannwhitney_vect1(self):
        vector_a = np.zeros(len(self.defaulted_customers))
        for i in range(len(self.defaulted_customers)):
            for j in range(len(self.nondefaulted_customers)):
                if self.ratings_defaulted[i] < self.ratings_nondefaulted[j]:
                    vector_a[i] = vector_a[i] + 1
                elif self.ratings_defaulted[i] == self.ratings_nondefaulted[j]:
                    vector_a[i] = vector_a[i] + 1/2
                else:
                    vector_a[i] = vector_a[i] + 0          
    
        return (vector_a/len(self.nondefaulted_customers))

    def mannwhitney_vect2(self):
        vector_b = np.zeros(len(self.nondefaulted_customers))
        for i in range(len(self.nondefaulted_customers)):
            for j in range(len(self.defaulted_customers)):
                if self.ratings_defaulted[j] < self.ratings_nondefaulted[i]:
                    vector_b[i] = vector_b[i] + 1
                elif self.ratings_defaulted[j] == self.ratings_nondefaulted[i]:
                    vector_b[i] = vector_b[i] + 1/2
                else:
                    vector_b[i] = vector_b[i] + 0          
    
        return (vector_b/len(self.defaulted_customers))
    
    def mannwhitney_var(self):
        vecta = self.mannwhitney_vect1()
        vectb = self.mannwhitney_vect2()
        sample_var_a = np.var(vecta, ddof=1)
        sample_var_b = np.var(vectb, ddof=1)
        x1 = sample_var_a / len(vecta)
        x2 = sample_var_b / len(vectb)
        return (x1 + x2)
    
    def test_statistic(self, auc_init: float):
        auc_curr = self.mannwhitney_auc()
        var = self.mannwhitney_var()
        S = (auc_init - auc_curr)/(np.sqrt(var))
        return S
    
    def summary_discr_power(self, auc_init = 0.3):
        fpr, tpr, thresholds = metrics.roc_curve(self.def_or_not, self.PD_score)
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        auc_curr = self.mannwhitney_auc()
        #auc_curr = metrics.roc_auc_score(self.def_or_not, self.PD_score)
        var = self.mannwhitney_var()
        S = self.test_statistic(auc_init)
        p_val = (1  - norm.cdf(S, loc=0, scale=1))
        summary_dict = {'Current auc': auc_curr, 'S metric': S, 'var':var, 'p-value':p_val}
        summary_dict = pd.DataFrame(columns= ['Current auc', 'S metric', 'var', 'p-value'])
        summary_dict.loc[0] = [auc_curr, S, var, p_val]
        return summary_dict, fig
        
        
    def migration_matrix_generation(self):
        
        row1_norm = [x / sum(self.row[0]) for x in self.row[0]]
        row2_norm = [x / sum(self.row[1]) for x in self.row[1]]
        row3_norm = [x / sum(self.row[2]) for x in self.row[2]]
        row4_norm = [x / sum(self.row[3]) for x in self.row[3]] 
        row5_norm = [x / sum(self.row[4]) for x in self.row[4]]
        row6_norm = [x / sum(self.row[5]) for x in self.row[5]]
        row7_norm = [x / sum(self.row[6]) for x in self.row[6]]
        
        migr_matrix = pd.DataFrame(np.array([row1_norm, row2_norm, row3_norm, row4_norm, row5_norm, row6_norm, row7_norm]), columns = self.ratings, index = self.prev_ratings)
        return migr_matrix    
    
    def M_norm_up(self):
        transitionmatrix = self.migration_matrix_generation()
        K = len(self.prev_ratings)
        total = 0
        for i in range(1, K):
            total = total + max(abs(i-K), abs(i-1)) * self.N_per_grade[i-1] * sum(transitionmatrix.iloc[i-1, i:K])
        return total
    
    def M_norm_low(self):
        transitionmatrix = self.migration_matrix_generation()
        K = len(self.prev_ratings)
        total = 0
        for i in range(2, K+1):
            total = total + max(abs(i-K), abs(i-1)) * self.N_per_grade[i-1] * sum(transitionmatrix.iloc[i-1 , 0: i-1])
        return total
    
    def upper_MWB(self):
        M_norm = self.M_norm_up()
        transitionmatrix = self.migration_matrix_generation()
        K = len(self.prev_ratings)
        total = 0
        for i in range(1, K):
            for j in range(i+1, K+1):
                total = total + (abs(i-j) * self.N_per_grade[i-1] * transitionmatrix.iloc[i-1, j-1])
        return (total / M_norm)
    
    def lower_MWB(self):
        M_norm = self.M_norm_low()
        transitionmatrix = self.migration_matrix_generation()
        K = len(self.prev_ratings)
        total = 0
        for i in range(2, K+1):
            for j in range(1, i):
                total = total + (abs(i-j) * self.N_per_grade[i-1] * transitionmatrix.iloc[i-1, j-1])
        return (total / M_norm)
    
    def z_test_matrix(self):
        K = len(self.prev_ratings)
        transitionmatrix = self.migration_matrix_generation()
        z_matrix = pd.DataFrame(index = np.array(transitionmatrix.index), columns = np.array(transitionmatrix.index))
        for i in range(1, K+1):
            for j in range(1, i):
                z_matrix.iloc[i-1,j-1] = (transitionmatrix.iloc[i-1,j] - transitionmatrix.iloc[i-1, j-1]) / np.sqrt(((transitionmatrix.iloc[i-1, j-1] * (1 - transitionmatrix.iloc[i-1, j-1])) / self.N_per_grade[i-1])+((transitionmatrix.iloc[i-1,j] * (1 - transitionmatrix.iloc[i-1,j])) / self.N_per_grade[i-1])+(2 * (transitionmatrix.iloc[i-1, j-1] * transitionmatrix.iloc[i-1,j]) / self.N_per_grade[i-1]))
            
        for i in range(1, K+1):
            for j in range(i+1, K+1):
                z_matrix.iloc[i-1, j-1] = (transitionmatrix.iloc[i-1, j-2] - transitionmatrix.iloc[i-1, j-1])/np.sqrt(((transitionmatrix.iloc[i-1, j-1] * (1 - transitionmatrix.iloc[i-1, j-1])) / self.N_per_grade[i-1])+((transitionmatrix.iloc[i-1, j-2] * (1 - transitionmatrix.iloc[i-1, j-2]))/ self.N_per_grade[i-1])+((2 * transitionmatrix.iloc[i-1, j-1] * transitionmatrix.iloc[i-1, j-2])/self.N_per_grade[i-1]))
        return z_matrix
    
    def p_value_matrix(self):
        z_matrix = self.z_test_matrix()
        p_matrix = pd.DataFrame(index = np.array(z_matrix.index), columns = np.array(z_matrix.index))
        for i in range(len(z_matrix)):
            for j in range(len(z_matrix)):
                if  np.isnan(z_matrix.iloc[i,j]) == True:
                    continue
                else:
                    p_matrix.iloc[i,j] = norm.cdf(z_matrix.iloc[i,j], loc=0, scale=1)
        return p_matrix
    
    def coef_of_variation_number(self):
        K = len(self.prev_ratings)
        R = np.zeros(K)
        for i in range(K):
            R[i] = R[i] + self.N_per_grade[i] / sum(self.N_per_grade)
        total = 0
        for j in range(K):
            total = total + ((R[j] - (1/K))**2)
    
        return np.sqrt(K * total)
    
    def coef_of_variation_exposure(self):
        K = len(self.prev_ratings)
        R = np.zeros(K)
        for i in range(K):
            R[i] = R[i] + self.exposure_per_grade[i] / sum(self.exposure_per_grade)
        total = 0
        for j in range(K):
            total = total + ((R[j] - (1/K))**2)
    
        return np.sqrt(K * total)
    
    def herfindahl_index_number(self):
        K = len(self.prev_ratings)
        cv_squared = self.coef_of_variation_number() ** 2
        h_index = 1 + (np.log((cv_squared + 1)/K)/np.log(K))
        return h_index
    
    def herfindahl_index_exposure(self):
        K = len(self.prev_ratings)
        cv_squared = self.coef_of_variation_exposure() ** 2
        h_index = 1 + (np.log((cv_squared + 1)/K)/np.log(K))
        return h_index
    
    def p_value_herfindahl_number(self, cv_initial: float):
        K = len(self.prev_ratings)
        cv_current = self.coef_of_variation_number()
        x = ((np.sqrt(K-1))*(cv_current - cv_initial))/(np.sqrt((cv_current**2) * (0.5 + (cv_current**2))))
        p_value = 1 - norm.cdf(x, loc=0, scale=1)
        return p_value
    
    def p_value_herfindahl_exposure(self, cv_initial: float):
        K = len(self.prev_ratings)
        cv_current = self.coef_of_variation_exposure()
        x = ((np.sqrt(K-1))*(cv_current - cv_initial))/(np.sqrt((cv_current**2) * (0.5 + (cv_current**2))))
        p_value = 1 - norm.cdf(x, loc=0, scale=1)
        return p_value
    
    def summary_stability(self, cv_initial: float):
        return("""The migration matrix is:\n{}
              \nThe summary statistics for the analysis of customer migrations is the Upper MWB and the Lower MWB. The values for these statistics are: \
              \nUpper MWB: {} \
              \nLower MWB: {}
              \nWith regard to the stability of the migration matrix, the z-matrix containing the z-test statistics is:\n{}
              \nThe matrix containing the p-values is:\n{}
              \nThe null hypothesis is either H_0: p_(i,j) bigger or equal to p_(i, j-1) or H_0: p_(i, j-1) bigger or equal to p_(i,j) \
              \ndepending on whether the (i,j) entry in the migration matrix is below or above the main diagonal resp.
              \nFor the concentration of rating grades we calculate the Herfindahl index, both at number-weighted level and at exposure-weighted level.
              \nThe number_weighted Herfindahl index is: {} \
              \nThe corresponding p-value with CV_(init) equal to {} is : {} 
              \nThe exposure-weighted Herfindahl index is {} \
              \nThe corresponding p-value with CV_(init) equal to {} is: {}
              \nThe null hypothesis is that the current Herfindahl index is lower than the Herfindahl index at the time of development. \
              \nTo provide a different initial Coefficient of Variation, provide it as input to the method.
              """.format(self.migration_matrix_generation(), self.upper_MWB(), self.lower_MWB(), self.z_test_matrix(),
              self.p_value_matrix(), self.herfindahl_index_number(), cv_initial, self.p_value_herfindahl_number(cv_initial), self.herfindahl_index_exposure(), 
              cv_initial, self.p_value_herfindahl_exposure(cv_initial)))
    
    def summary_stability_2(self, cv_initial: float):
        return (self.migration_matrix_generation(), self.upper_MWB(), self.lower_MWB(), self.z_test_matrix(),
              self.p_value_matrix(), self.herfindahl_index_number(), self.p_value_herfindahl_number(cv_initial), 
              self.herfindahl_index_exposure(), self.p_value_herfindahl_exposure(cv_initial))
        
    