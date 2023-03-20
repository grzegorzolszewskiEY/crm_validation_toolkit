import pandas as pd
import numpy as np
from scipy.stats import t
import warnings
from t_test_assumptions import t_test_assumptions

warnings.filterwarnings('ignore')


class LGD_Validator:
    def __init__(self):
        pass

    # funkcja tworzaca buckety
    def segmentation(self, df: pd.DataFrame, segmentation_column: str) -> pd.DataFrame:
        bins = [i / 10 for i in range(11)]
        bins.extend([0.05, np.inf])
        bins = np.array(sorted(bins))

        data = df.copy().sort_values(by=segmentation_column)
        data['bucket'] = pd.cut(df[segmentation_column], bins, right=False)
        return data

    def elbe_t_test(self, n, lgdr, elbe):
        assert n == len(lgdr) == len(elbe)
        if n < 2:
            n = 2

        denominator = sum(lgdr - elbe)
        if denominator == 0:
            denominator = 0.001

        t_test_assumption = t_test_assumptions()
        test_assumptions_met = t_test_assumption.assumptions_met(lgdr, elbe, alpha = 0.05)

        eq_part = 1 / n * denominator
        s_squared = sum((lgdr - elbe - eq_part) ** 2) / (n - 1)
        t_stat = np.sqrt(n) * eq_part / np.sqrt(s_squared)
        p_value = 2 * (1 - t.cdf(abs(t_stat), n - 1))
        return t_stat, s_squared, p_value, test_assumptions_met

    def lgdd_t_test(self, n, lgdr, lgdd):
        assert n == len(lgdr) == len(lgdd)
        if n < 2:
            n = 2

        denominator = sum(lgdr - lgdd)
        if denominator == 0:
            denominator = 0.001

        ###edited ~Filip###
        t_test_assumption = t_test_assumptions()
        test_assumptions_met = t_test_assumption.assumptions_met(lgdr, lgdd, alpha = 0.05)
      
        eq_part = 1 / n * denominator
        s_squared = sum((lgdr - lgdd - eq_part) ** 2) / (n - 1)
        t_stat = np.sqrt(n) * eq_part / np.sqrt(s_squared)
        p_value = 1 - t.cdf(t_stat, n - 1)
        return t_stat, s_squared, p_value, test_assumptions_met

    def lgd_defaulted_portfolio_v1(self, data, observation_start_date, observation_end_date, beginning_col, end_col):
        # defaulted facilities - recovery process not closed
        # defaulted in our observation period

        df = data.copy()

        beginning_of_period = pd.to_datetime(observation_start_date)
        end_of_period = pd.to_datetime(observation_end_date)

        df[beginning_col] = pd.to_datetime(df[beginning_col])
        df[end_col] = pd.to_datetime(df[end_col])
        df['default_in_days'] = (df[end_col] - df[beginning_col]).dt.days

        # najpierw kalkulacje na poczatku roku - facilities defaulted dnia obs_start_date
        df_beginning = df[(df[beginning_col] < beginning_of_period) & (beginning_of_period < df[end_col])]
        rwea_beginning = df_beginning['RWEA_0y_after_default']
        exposure_beginning = df_beginning['exposure_0y_after_default']
        elbe_beginning = df_beginning['ELBE_0y_after_default']

        output_beginning = pd.DataFrame(columns=['average RWEA', 'RWEA total sum', 'exposure total sum',
                                                 'average exposure', 'average ELBE',
                                                 'number of facilities', 'average days in default'])

        output_beginning.loc[0] = [int(np.average(rwea_beginning)), sum(rwea_beginning),
                                   sum(exposure_beginning), int(np.average(exposure_beginning)),
                                   np.average(elbe_beginning), df_beginning.shape[0],
                                   int(np.average(df_beginning['default_in_days']))]

        # kalkulacje na koncu roku - facilities defaulted dnia obs_end_date
        df_end = df[(df[beginning_col] < end_of_period) & (end_of_period < df[end_col])]
        rwea_end = df_end['RWEA_0y_after_default']
        exposure_end = df_end['exposure_0y_after_default']
        elbe_end = df_end['ELBE_0y_after_default']

        output_end = pd.DataFrame(columns=['average RWEA', 'average exposure', 'average ELBE',
                                           'number of facilities', 'average days in default'])

        output_end.loc[0] = [int(np.average(rwea_end)), int(np.average(exposure_end)),
                             np.average(elbe_end), df_end.shape[0],
                             np.average(df_end['default_in_days']).astype(int)]

        return output_beginning, output_end

    def lgd_defaulted_portfolio_v2(self, data, observation_start_date, observation_end_date, beginning_col, end_col):
        # facilities with recovery process closed within observation period
        # avg days in def for facilities which closed default in CERTAIN YEAR

        df = data.copy()

        beginning_of_period = pd.to_datetime(observation_start_date)
        end_of_period = pd.to_datetime(observation_end_date)

        df[beginning_col] = pd.to_datetime(df[beginning_col])
        df[end_col] = pd.to_datetime(df[end_col])
        df['default_in_days'] = (df[end_col] - df[beginning_col]).dt.days
        df = df[(beginning_of_period <= df[end_col]) & (df[end_col] <= end_of_period)]

        output = pd.DataFrame(columns=[f'Average days in default for customers with recovery proces finished between '
                                       f'{observation_start_date} and {observation_end_date}'])
        output.loc[0] = round(np.average(df['default_in_days']))

        return output

    def elbe_t_test_report(self, data, observation_start_date, observation_end_date, end_col, alpha):
        # robie test na podstawie kolumn elbe_at_def, elbe_1y_after_def itd
        # biorac pod uwage tylko klientow ktorzy skonczyli def w danym roku

        df = data.copy()

        beginning_of_period = pd.to_datetime(observation_start_date)
        end_of_period = pd.to_datetime(observation_end_date)

        df[end_col] = pd.to_datetime(df[end_col])
        df = df[(beginning_of_period <= df[end_col]) & (df[end_col] <= end_of_period)]

        years_after_def = [0, 1, 3, 5, 7]
        report_dataframes = []

        for i in years_after_def:
            elbe_col = 'ELBE_' + str(i) + 'y_after_default'
            lgdd_col = 'LGD_in-default_' + str(i) + 'y_after_default'
            lgdr_col = 'LGD_realised_' + str(i) + 'y_after_default'

            df_period = self.segmentation(df.copy(), elbe_col)
            df_period = df_period[~df_period[elbe_col].isnull()]

            output = pd.DataFrame(columns=['bucket', 'number of facilities', 'ELBE average', 'LGDr average',
                                           'LGDd average', 't statistic', 's squared', 'p-value', 't-test_assumption_met',
                                           f'is H0 rejected for alpha={alpha}'])

            for idx, bucket in enumerate(df_period['bucket'].unique()):
                lgdr_bucket, elbe_bucket, lgdd_bucket = \
                    (df_period[df_period['bucket'] == bucket])[lgdr_col], \
                    (df_period[df_period['bucket'] == bucket])[elbe_col], \
                    (df_period[df_period['bucket'] == bucket])[lgdd_col]

                t_stat, s_squared, p_value, test_assumptions_met = self.elbe_t_test(len(lgdr_bucket), lgdr_bucket, elbe_bucket)

                new_row = [bucket, len(lgdr_bucket), np.average(elbe_bucket), np.average(lgdr_bucket),
                           np.average(lgdd_bucket), t_stat, s_squared, round(p_value, 5), test_assumptions_met, p_value < alpha]

                output.loc[idx + 1] = new_row

            t_stat, s_squared, p_value, test_assumptions_met = self.elbe_t_test(len(df_period[lgdr_col]), df_period[lgdr_col],
                                                          df_period[elbe_col])

            new_row = ['Portfolio level', len(df_period[lgdr_col]), np.average(df_period[elbe_col]),
                       np.average(df_period[lgdr_col]), np.average(df_period[lgdd_col]), t_stat,
                       s_squared, round(p_value, 5), test_assumptions_met, p_value <= alpha]

            output.loc[13] = new_row

            if len(df_period[lgdr_col]) > 0:
                report_dataframes.append(output)
            else:
                report_dataframes.append(f'There are no facilities that are in default for {i} years, '
                                         f'for which recovery process closed between {observation_start_date} and {observation_end_date}.')

        return report_dataframes

    def lgdd_t_test_report(self, data, observation_start_date, observation_end_date, end_col, alpha):
        df = data.copy()

        beginning_of_period = pd.to_datetime(observation_start_date)
        end_of_period = pd.to_datetime(observation_end_date)

        df[end_col] = pd.to_datetime(df[end_col])
        df = df[(beginning_of_period <= df[end_col]) & (df[end_col] <= end_of_period)]

        years_after_def = [0, 1, 3, 5, 7]
        report_dataframes = []

        for i in years_after_def:
            elbe_col = 'ELBE_' + str(i) + 'y_after_default'
            lgdd_col = 'LGD_in-default_' + str(i) + 'y_after_default'
            lgdr_col = 'LGD_realised_' + str(i) + 'y_after_default'

            df_period = self.segmentation(df.copy(), lgdd_col)
            df_period = df_period[~df_period[lgdd_col].isnull()]

            output = pd.DataFrame(columns=['bucket', 'number of facilities', 'LGDd average', 'LGDr average',
                                           'ELBE average', 't statistic', 's squared', 'p-value', 't-test_assumption_met',
                                           f'is H0 rejected for alpha={alpha}'])

            for idx, bucket in enumerate(df_period['bucket'].unique()):
                lgdr_bucket, elbe_bucket, lgdd_bucket = \
                    (df_period[df_period['bucket'] == bucket])[lgdr_col], \
                    (df_period[df_period['bucket'] == bucket])[elbe_col], \
                    (df_period[df_period['bucket'] == bucket])[lgdd_col]

                t_stat, s_squared, p_value, test_assumptions_met = self.lgdd_t_test(len(lgdr_bucket), lgdr_bucket, lgdd_bucket)

                new_row = [bucket, len(lgdr_bucket), np.average(lgdd_bucket), np.average(lgdr_bucket),
                           np.average(elbe_bucket), t_stat, s_squared, round(p_value, 5), test_assumptions_met, p_value < alpha]

                output.loc[idx + 1] = new_row

            t_stat, s_squared, p_value, test_assumptions_met = self.lgdd_t_test(len(df_period[lgdr_col]), df_period[lgdr_col],
                                                          df_period[lgdd_col])

            new_row = ['Portfolio level', len(df_period[lgdr_col]), np.average(df_period[lgdd_col]),
                       np.average(df_period[lgdr_col]), np.average(df_period[elbe_col]), t_stat,
                       s_squared, round(p_value, 5), test_assumptions_met, p_value <= alpha]

            output.loc[13] = new_row

            if len(df_period[lgdr_col]) > 0:
                report_dataframes.append(output)
            else:
                report_dataframes.append(f'There are no facilities that are in default for {i} years, '
                                         f'for which recovery process closed between {observation_start_date} and {observation_end_date}.')
        return report_dataframes
