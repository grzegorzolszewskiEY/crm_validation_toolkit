from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom, norm, uniform

from crm_validator.util.gAUC_class import gAUC
from crm_validator.lgd.LGDvalidator import LGDValidator
from crm_validator.lgd.LGD_validator import LGD_Validator
from crm_validator.pd.PDvalidation import PD_validator


def pd_lgd_app(st, uploaded_files):
    #### Validators ####
    first_validator = LGDValidator()
    second_validator = LGD_Validator()

    st.title("LGD Validator")

    st.write("The application performs validation in accordance with the guidelines contained in the document: [link](https://www.bankingsupervision.europa.eu/banking/tasks/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf)")

    st.markdown("---")
    input_file_PD = None 
    input_file_LGD = None
    input_file_CCF = None

    #uploaded_files = st.file_uploader('Upload files', accept_multiple_files= True)
    for uploaded_file in uploaded_files:

        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name.upper())

        if "LGD" in uploaded_file.name.upper():  #CHANGE IT TO "LGD" NOT "reports"
            uploaded_file.seek(0)  #Since pd.read_csv depletes the buffer, the second time you call read_csv will return no rows.
            input_file_LGD = uploaded_file
        elif "PD" in uploaded_file.name.upper():
            input_file_PD = uploaded_file
        elif "APP" in uploaded_file.name.upper():
            uploaded_file.seek(0)
            application_portfolio = pd.read_csv(uploaded_file)
        else:
            input_file_CCF = uploaded_file

    if input_file_LGD and application_portfolio is not None:
        relevant_obs_per_beg = st.text_input('Beginning of relevant observation period', '2015-01-01')
        relevant_obs_per_end = st.text_input('End of relevant observation period', '2016-01-01')

    st.markdown("---")

    if input_file_PD is not None:
        input_file_PD = pd.read_excel(input_file_PD)
        #input_file_PD_for_quali = pd.read_excel(input_file_PD)
        PD_valid = PD_validator(input_file_PD) #initialize PD_validator class

        ### PD Validaiton part ###
        jeff_test_df = PD_valid.summary_pred_ability(0.05)

        quali_inf = PD_valid.qualitative_validation1(input_file_PD)
        quali_inf_2 = PD_valid.qualitative_validation2(input_file_PD)
        
        with st.expander('PD back-testing using a Jeffreys test'):
            st.write('Jeffreys test both at individual grade level and at portfolio level.') 
            st.write('The null hypothesis is that the estimated PD is greater than the true one.')
            st.dataframe(jeff_test_df)

        with st.expander('PD discriminatory power'):    
            init_AUC = st.number_input('Insert a initial AUC', value =0.3)
            summ_disc_df = PD_valid.summary_discr_power(init_AUC)
            st.write('ROC Curve')
            st.pyplot(summ_disc_df[1])
            st.write('Current AUC vs. AUC at initial validation/development')
            st.dataframe(summ_disc_df[0])

        with st.expander('PD qualitative validation tools'):
            st.write(quali_inf[0], quali_inf[1])
            st.write(quali_inf[2], quali_inf[3])
            st.write(quali_inf[4], quali_inf[5])
            st.write(quali_inf_2[0], quali_inf_2[1])
            
        with st.expander('PD stability'):
            cv_initial = st.number_input('Insert a initial CV', value =0.1)
            summ_stab_df = PD_valid.summary_stability_2(cv_initial)

            st.write('The migration matrix is:')
            st.dataframe(summ_stab_df[0])
            st.write('''The summary statistics for the analysis of customer migrations is the Upper MWB and the Lower MWB. 
                    The values for these statistics are:''')
            st.write('Upper MWB:', summ_stab_df[1])
            st.write('Lower MWB:', summ_stab_df[2])
            st.write('''With regard to the stability of the migration matrix, the z-matrix containing the z-test statistics is:''')
            st.dataframe(summ_stab_df[3])
            st.write('''The matrix containing the p-values is:''')
            st.dataframe(summ_stab_df[4])
            st.write('''The null hypothesis is either H_0: p_(i,j) bigger or equal to p_(i, j-1) or H_0: p_(i, j-1) bigger or equal to p_(i,j) depending on whether the (i,j) entry in the migration matrix is below or above the main diagonal resp.
                    For the concentration of rating grades we calculate the Herfindahl index, both at number-weighted level and at exposure-weighted level.''')
            st.write('''The number_weighted Herfindahl index is:''', summ_stab_df[5])
            st.write('''The corresponding p-value with CV_(init) equal to {} is:'''.format(cv_initial), summ_stab_df[6])
            st.write('')
            st.write('''The exposure-weighted Herfindahl index is''', summ_stab_df[7])
            st.write('''The corresponding p-value with CV_(init) equal to {} is:'''.format(cv_initial), summ_stab_df[8])
            st.write('''The null hypothesis is that the current Herfindahl index is lower than the Herfindahl index at the time of development.               
                    To provide a different initial Coefficient of Variation, provide it as input to the method.''')


    if input_file_LGD and application_portfolio is not None:

        ### LGD Validation part ###
        LGD_data = pd.read_csv(input_file_LGD)
        required_idx = first_validator.pick_selected_obs(LGD_data, relevant_obs_per_beg, relevant_obs_per_end, 'end_of_default')
        LGD_data_slice = LGD_data.loc[required_idx]
        segmentated_data = first_validator.segmentation(df = LGD_data_slice, segment_name = 'Estimated_LGD_Bucket', target = 'LGD_estimated')
        segmentated_data = first_validator.segmentation(df = segmentated_data, segment_name = 'Realised_LGD_Bucket', target = 'LGD_realised_0y_after_default')

        LGD_back_df = first_validator.t_test_LGD_each_segment(data=segmentated_data, segment_col_name= 'Estimated_LGD_Bucket',LGDr_col= 'LGD_realised_0y_after_default', 
                                    LGDe_col= 'LGD_estimated')

        matrix = first_validator.contigency_table(df = segmentated_data, estimated_segment_col= 'Estimated_LGD_Bucket', real_segment_col= 'Realised_LGD_Bucket')
        LGD_back_contigency = first_validator.contigency_table_df(matrix)

        gAUC =  gAUC_class.gAUC()
        curr_gAUC = gAUC.calc_gAUC(matrix)
        curr_std = gAUC.standard_dev(matrix)
        var_std = curr_std**2
        gAUC_metric_df = pd.DataFrame(columns= ['current gAUC', 'std', 'var'])
        gAUC_metric_df.loc[0] = [curr_gAUC, curr_std, var_std]

        #required_idx = first_validator.pick_selected_obs_non_def(LGD_data, relevant_obs_per, 'beginning_of_default', 'end_of_default', beg = True)
        #LGD_data_slice = LGD_data.loc[required_idx]
        miss_val, miss_plot = first_validator.miss_prop(data = application_portfolio, def_col = 'forced_or_missing', miss_col= 'forced_or_missing')
        miss_val_df = pd.DataFrame(columns= ['%_of_missing_values'])
        miss_val_df.loc[0] = [miss_val]

        application_portfolio = application_portfolio[application_portfolio['forced_or_missing'] != 1] #exclude missing values

        chosen_idx_beg = list(np.sort((np.random.choice(application_portfolio.shape[0], replace=False,
                                                        size=np.random.randint(application_portfolio.shape[0]/2, application_portfolio.shape[0])))))
        chosen_idx_end = list(np.sort(np.random.choice(application_portfolio.shape[0], replace=False, 
                                                    size=np.random.randint(application_portfolio.shape[0]/2, application_portfolio.shape[0]))))

        segmentated_data_beg = first_validator.segmentation(df = application_portfolio.iloc[chosen_idx_beg], 
                                                            segment_name = 'Estimated_LGD_Bucket', target = 'LGD_estimated')
        app_portfolio_beg_df = first_validator.app_port_distibution(segmentated_data_beg, 'Estimated_LGD_Bucket',  LGDe_col= 'LGD_estimated',
                        collateral_name= 'Col_rate', original_exp= 'exposure')
        
        segmentated_data_end = first_validator.segmentation(df = application_portfolio.iloc[chosen_idx_end], 
                                                            segment_name = 'Estimated_LGD_Bucket', target = 'LGD_estimated')
        app_portfolio_end_df = first_validator.app_port_distibution(segmentated_data_end, 'Estimated_LGD_Bucket',  LGDe_col= 'LGD_estimated',
                        collateral_name= 'Col_rate', original_exp= 'exposure')

        population_comparison = first_validator.PSI_preparation(df_initial = segmentated_data_beg, df_curr = segmentated_data_end,
        segment_col_initial= 'Estimated_LGD_Bucket', segment_col_curr = 'Estimated_LGD_Bucket')
        PSI = population_comparison['Contribution'].sum()
        psi_val_df = pd.DataFrame(columns= ['PSI'])
        psi_val_df.loc[0] = [PSI]  

        lgd_def_not_closed_df = second_validator.lgd_defaulted_portfolio_v1(LGD_data, relevant_obs_per_beg, 
                                                                            relevant_obs_per_end, 'beginning_of_default', 'end_of_default')[0]

        lgd_def_not_closed_df_two = second_validator.lgd_defaulted_portfolio_v1(LGD_data, relevant_obs_per_beg, 
                                                                                relevant_obs_per_end, 'beginning_of_default', 'end_of_default')[1]

        lgd_def_colsed_within_df = second_validator.lgd_defaulted_portfolio_v2(LGD_data, relevant_obs_per_beg, 
                                                                            relevant_obs_per_end, 'beginning_of_default', 'end_of_default')
        
        outputs_elbe = second_validator.elbe_t_test_report(LGD_data, relevant_obs_per_beg, relevant_obs_per_end, 'end_of_default', 0.05)

        outputs_lgdd = second_validator.lgdd_t_test_report(LGD_data, relevant_obs_per_beg, relevant_obs_per_end, 'end_of_default', 0.05)

        with st.expander('LGD back-testing using a t-test'):
            st.write('Summary statistics')
            st.dataframe(LGD_back_df)
            st.write('Contigency table (rows represent estimated LGD segment and columns represent relised LGD segment)')
            st.dataframe(LGD_back_contigency)

        with st.expander('LGD discriminatory power'):
            st.write('Current gAUC')
            st.dataframe(gAUC_metric_df)

            st.write('gAUC current and initial')
            init_gAUC = st.number_input('Insert a initial gAUC', value=0.5)
            S_metric_df = first_validator.normal_test(curr_gAUC, init_gAUC, curr_std)
            st.dataframe(S_metric_df)

            st.write('Contigency table (rows represent estimated LGD segment and columns represent realised LGD segment)')
            st.dataframe(LGD_back_contigency)

        with st.expander('LGD qualitative validation tools'):
            st.write('Missing values at the beginning of relevant observation period')
            st.dataframe(miss_val_df)
            st.pyplot(miss_plot)
            st.write('Beginning of the relevant observation period (application portfolio)')
            st.dataframe(app_portfolio_beg_df)
            st.write('End of the relevant observation period (application portfolio)')
            st.dataframe(app_portfolio_end_df)
            st.write('Population stability index (PSI)')
            st.dataframe(psi_val_df)
            st.write('LGD defaulted portfolio (recovery process not closed)')
            st.markdown('All facilities that were in default at the beginning of the observation period')
            st.dataframe(lgd_def_not_closed_df)
            st.write('All facilities that were in default at the end of the observation period')
            st.dataframe(lgd_def_not_closed_df_two)
            st.write('LGD defaulted portfolio (recovery process closed within the observation period)')
            st.dataframe(lgd_def_colsed_within_df)

        with st.expander('ELBE backtesting using t-test'):
            st.write('For the relevant year reports has been created for ELBE at the moment of default and one, three, five, seven years after default.')
            if type(outputs_elbe[0]) == str:
                st.write(outputs_elbe[0])
            else:
                st.dataframe(outputs_elbe[0])

            for i in [0, 1, 2, 3, 4]:
                if type(outputs_elbe[i]) == str:
                    st.write(outputs_elbe[i])
                else:
                    st.dataframe(outputs_elbe[i])
        
        with st.expander('LGD in-default backtesting using t-test'):
            for i in [0, 1, 2, 3, 4]:
                if type(outputs_lgdd[i]) == str:
                    st.write(outputs_lgdd[i])
                else:
                    st.dataframe(outputs_lgdd[i])
                    