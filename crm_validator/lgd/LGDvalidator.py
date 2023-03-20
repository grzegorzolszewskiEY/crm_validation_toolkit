import numpy as np
import pandas as pd 
from scipy.stats import uniform, binom, norm, beta, t
from crm_validator.util.gAUC_class import gAUC
from crm_validator.util.t_test_assumptions import t_test_assumptions
import matplotlib.pyplot as plt

class LGDValidator():

    def ___init___(self) -> None:
        pass

    def add_data(self, data: pd.DataFrame):
        self.data = data
        return self.data

    #testing predictive ability
    def t_test_LGD(self, LGDr: np.array, LGDe: np.array):
        '''
        N - number of customers, it is calcualted in function as length of LGDr
        LGDr - Realised Loss given default (np.array type)
        LGDe - Estimated Loss given default (np.array type)
        '''
        N = len(LGDr)  
        eq_part = 1/len(LGDr) * sum(LGDr - LGDe)
        S_sqrt = sum((LGDr - LGDe - eq_part)**2) / (N-1)
        T = np.sqrt(N) * eq_part / np.sqrt(S_sqrt)
        p_value = 1 - t.cdf(T, N-1)
        return (N, p_value, T, S_sqrt)

    def segmentation(self, df: pd.DataFrame, segment_name: str, target: str):

        '''
            df: pd.Dataframe type of data with realised and estimeted LGD
            segment_name: segmentated data are store in new column called segment_name
            target: user specified with part should be segmentated
        '''
    
        df = df.copy()
        df[segment_name] = [0] * df.shape[0]
        df[segment_name] = np.where((df[target] >= 0) & (df[target] < 0.05), df[segment_name] + 1, df[segment_name])
        df[segment_name] = np.where((df[target] >= 0.05) & (df[target] < 0.1), df[segment_name] + 2, df[segment_name])

        for i in range(2, 11):
            df[segment_name] = np.where((df[target] >= 0.1*(i-1)) & (df[target] < 0.1*i), df[segment_name] + (i+1), df[segment_name])

        df[segment_name] = np.where(df[target] >= 1, df[segment_name] + 12, df[segment_name])
        return df

    def contigency_table(self, df: pd.DataFrame, estimated_segment_col: str, real_segment_col: str):
        '''
            df: pd.DataFrame which contains columns with corresponding buckets of realised and estimated LGD's
            estimated_segment_col: name of column in df which contains buckets of estimated LGD
            real_segment_col: name of column in df which contains buckets of realised LGD

            IMPORTANT: funciton returns contigency table for selected columns in df, rows are represented as
            estimated_segment_col and columns as real_segment_col, it means that rows are estimated LGD buckets
            and columns are realised.
        '''
        matrix = np.zeros(shape=(12, 12), dtype=object) #previously dtype = int64, change to object to store big numbers 

        for pair in zip(df[estimated_segment_col], df[real_segment_col]):
            #segments are indexed from one while in Python index starts from zero so (1,1) coressponds to (0,0) in matrix
            matrix[pair[0]-1,pair[1]-1] = matrix[pair[0]-1,pair[1]-1] + 1
        return matrix 

    def contigency_table_df(self, matrix: np.array):
        '''
            matrix: contigency table
            Function returns dataframe from given contigency table with indexes move by one
            rows are represented as estimated_segment_col and columns as real_segment_col
        '''
        table = pd.DataFrame(data=matrix)
        #add +1 to index and column to get range from 1 to 12 insted of 0 to 11
        table.index = np.arange(1, len(table) + 1)
        table.columns = np.arange(1, len(table) + 1)
        return table

    def select_segment(self, data: pd.DataFrame, segment_col_name: str, segment: int):
        '''
            data: pd.Dataframe which contains segmentated LGD's
            segment_col_name: name of column where information about segments is stored
            segment: number of selected segment
        
        '''
        return data[data[segment_col_name]==segment].reset_index(drop = True)

    def t_test_LGD_each_segment(self, data: pd.DataFrame, segment_col_name: str, LGDr_col: str, LGDe_col: str, alpha = 0.05):
        '''
            Function performs t_test_LGD for each segment in given dataframe (data) and selected column (segment_col_name)
        '''
        results_df = pd.DataFrame(columns=['segment', 'N', 't stat', 's squared', 'p-value', 'LGDr_avg', 'LGDe_avg', 't-test_assumptions_met'])
        t_test_assumption = t_test_assumptions()

        for i in np.sort(data[segment_col_name].unique()):
            temp_df = self.select_segment(data, segment_col_name, i)
            LGDr = np.array(temp_df[LGDr_col].tolist())
            LGDe = np.array(temp_df[LGDe_col].tolist())
            LGDr_avg = temp_df[LGDr_col].mean()
            LGDe_avg = temp_df[LGDe_col].mean()
            test_assumptions_met = t_test_assumption.assumptions_met(LGDr, LGDe, alpha)
            N, p_value, T, S_sqrt = self.t_test_LGD(LGDr, LGDe)
            new_row = pd.Series({'segment':int(i), 'N':int(N), 't stat':T, 's squared':S_sqrt,
            'p-value':p_value, 'LGDr_avg':LGDr_avg, 'LGDe_avg':LGDe_avg, 't-test_assumptions_met': test_assumptions_met})
            results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)
        results_df[['N']] = results_df[['N']].astype(int) #change column type to int

        #add part responsible for portfolio level calculations
        LGDr_avg = data[LGDr_col].mean()
        LGDe_avg = data[LGDe_col].mean()
        test_assumptions_met = t_test_assumption.assumptions_met(np.array(data[LGDr_col].tolist()), np.array(data[LGDe_col].tolist()), alpha)
        N, p_value, T, S_sqrt = self.t_test_LGD(LGDr = np.array(data[LGDr_col].tolist()), LGDe = np.array(data[LGDe_col].tolist()))
        new_row = pd.Series({'segment':'Portfolio level', 'N':int(N), 't stat':T, 's squared':S_sqrt, 
        'p-value':p_value, 'LGDr_avg':LGDr_avg, 'LGDe_avg':LGDe_avg, 't-test_assumptions_met': test_assumptions_met})
        results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)
        results_df[f'is H0 rejected for alpha={alpha}'] = results_df['p-value'] < alpha
        return results_df

    def normal_test(self, curr_gAUC: float, init_gAUC: float, curr_std: float, alpha = 0.05):
        S_metric = (init_gAUC - curr_gAUC) / curr_std
        test = 1 - norm.cdf(S_metric)
        S_metric_df = pd.DataFrame(columns= ['S metric', 'p-value'])
        S_metric_df.loc[0] = [S_metric, test]
        S_metric_df[f'is H0 rejected for alpha={alpha}'] = S_metric_df['p-value'] < alpha
        return S_metric_df
    
    def PSI_preparation(self, df_initial: pd.DataFrame, df_curr: pd.DataFrame, segment_col_initial: str, segment_col_curr: str):
        df_temp = df_initial
        core = df_temp.value_counts([segment_col_initial]) / df_temp.shape[0]
        df_temp = df_curr
        core_2 = df_temp.value_counts([segment_col_curr]) / df_temp.shape[0]
        dff = pd.concat([core, core_2], axis = 1)
        dff.sort_index(inplace=True)
        dff.columns = ['begin', 'end']
        dff['Contribution'] = np.where((dff['begin'] == 0) |(dff['end'] == 0), 0,
        (dff['end'] - dff['begin']) * np.log(dff['end']/dff['begin']))
        return dff

    def app_port_distibution(self, data: pd.DataFrame, segment_col_name: str, LGDe_col: str, 
                        collateral_name: str, original_exp: str):
        '''
            function returns: avg. of LGD estimated and collateralisation rate for each segment
            Additionally number of facilities and sum of original exposure (for each segment) will be returned 
        '''
        results_df = pd.DataFrame(columns=['segment', 'N', 'avg_estimated_LGD', 'avg_coll_rate', 'original_exposure'])
        
        for i in np.sort(data[segment_col_name].unique()):
            temp_df = self.select_segment(data, segment_col_name, i)

            LGDe_mean = np.array(temp_df[LGDe_col].tolist()).mean()
            coll_mean = np.array(temp_df[collateral_name].tolist()).mean()
            original_exposure = np.array(temp_df[original_exp].tolist()).sum()
            N = len(temp_df[LGDe_col].tolist())

            new_row = pd.Series({'segment':int(i), 'N':int(N), 'avg_estimated_LGD':LGDe_mean, 'avg_coll_rate':coll_mean, 
                                'original_exposure':original_exposure})
            results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)
        results_df[['N']] = results_df[['N']].astype(int) #change column type to int
        return results_df

    def miss_prop(self, data: pd.DataFrame, def_col: str, miss_col: str):
        '''
            miss_prop function returns ratio of rows marked as missing to rows marked as non-defaulted
            data - dataframe which contains columns with missing/non-defaulted status information
            def_col - name of column with default status information
            miss_col - name of column with missing/forced status information
        '''
        # m_summary non-defaulted facilities
        m_summary = data[data[def_col] == 0].shape[0]
        # m_miss faciliteis with missing or forced values
        m_miss = data[data[miss_col] == 1].shape[0]
        dict_for_plot = {'Missing/Forced': m_miss, 'Not Missing/Forced': m_summary}
        fig = plt.figure()
        plt.pie(dict_for_plot.values(), labels=dict_for_plot.keys())
        return m_miss/m_summary, fig

    def pick_selected_obs(self, dataset: pd.DataFrame, relevant_obs_per_beg: int, relevant_obs_per_end: int, end_def_col: str):
        '''
            relevant_obs_per_beg - beginning o frelevant observation period
            relevant_obs_per_end - 
            end_def_col - column name in dataset which contains information about date of default status ends.
            pick_selected_obs function returns indexes of observations in dataset for which default state ends 
            within relevant observation period 
        '''
        true_index = 0
        referance_date_beg = pd.to_datetime(relevant_obs_per_beg)
        referance_date_end = pd.to_datetime(relevant_obs_per_end)
        required_idx = []

        for idx, row in dataset.iterrows():
            
            if pd.to_datetime(row[end_def_col]) > referance_date_beg and pd.to_datetime(row[end_def_col]) < referance_date_end:
                required_idx.append(idx)
            
        return required_idx

    def pick_selected_obs_non_def(self, dataset: pd.DataFrame, relevant_obs_per: int, beg_def_col: str, end_def_col: str, beg = True):
        '''
            relevant_obs_per - year represents observation period for example 2016
            beg\end_def_col - column name in dataset which contains information about date of default status begins\ends.
            beg - declarate which point in time should be considered, if true: relevant_obs_per - 01 - 01 will be considered
            pick_selected_obs_non_def function returns indexes of observations in dataset for which default not occur  
            within relevant observation period 
        '''
        if beg:
            referance_date = str(relevant_obs_per) + '-01-01'
            referance_date = pd.to_datetime(referance_date)
        if not beg:
            referance_date = str(relevant_obs_per) + '-12-31'
            referance_date = pd.to_datetime(referance_date)
        required_idx = []
        for idx, row in dataset.iterrows():
            
            if pd.to_datetime(row[beg_def_col]) > referance_date and pd.to_datetime(row[end_def_col]) > referance_date or \
            pd.to_datetime(row[beg_def_col]) < referance_date and pd.to_datetime(row[end_def_col]) < referance_date:
                required_idx.append(idx)
            
        return required_idx