import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pandas.io import sql
import datetime
from IPython.core.debugger import set_trace
import subprocess


# #### MAIN CODE
# 
# #### SELECTING DATA
# 
# 1. Replace Earning reporting date ('rdq') = null rows with rdq = datadate + 45 days (assuming 45 day delay between
# quarter end & earning release)
# 2. Quarters start from 196206 (second calendar quarter) end 201712 (fourth calendar quarter)

def load_sp500_constituent_data(filepath = 'sp500_constituents.csv', cur_qtr = 20171231):
    members_sp500 = pd.read_csv(filepath)
    members_sp500.loc[pd.isnull(members_sp500['thru']), 'thru'] = 20171231
    return members_sp500


def get_sp500_constituents(calendar_qtr = 201712, members_sp500=None):
    if members_sp500 is None:
        members_sp500 = load_sp500_constituent_data('sp500_constituents.csv')
        
    t_mask = np.logical_and(members_sp500['from'] <= calendar_qtr*100+30, members_sp500['thru'] >= calendar_qtr*100+30)
    return list(members_sp500.loc[t_mask, 'gvkey'].unique())

# sp500_const_dict = {}
# members_sp500 = load_sp500_constituent_data('sp500_constituents.csv')
# const_count_dict = {}

# for cal_yr in np.arange(1964, 2017, 1):
    # for cal_month in [3, 6, 9, 12]:
        # cal_qtr = cal_yr*100+cal_month
        # t_const = get_sp500_constituents(calendar_qtr=cal_qtr, members_sp500=None)
        # sp500_const_dict[cal_qtr] = t_const
        # const_count_dict[cal_qtr] = len(t_const)


def fillna_rdq(int_date, offset=90):
    x = pd.to_datetime(int_date, format='%Y%m%d') + pd.DateOffset(days=offset)
    return int(x.strftime(format='%Y%m%d'))


def load_co_data(file_path='sp500_companies.csv'):
    sp500 = pd.read_csv(file_path)
    t_series = sp500.loc[:, 'datadate'].apply(lambda x: fillna_rdq(x, offset=90))

    sp500.loc[:,'rdq'] = sp500.loc[:,'rdq'].fillna(t_series)
    sp500.loc[:,'data_avail_date'] = sp500.loc[:, 'datadate'].apply(lambda x: fillna_rdq(x, offset=90))
    return sp500


def get_co_data(data_df, cal_qtr=200203, num_periods=1, verbose=False):
    ## data_df must have columns gvkey, datadate, data_avail_date
    
    month_day_dict = {3: 31, 6: 30, 9: 30, 12:31}

    cal_month = cal_qtr%100
    cal_year = int(cal_qtr/100)

    prev_month = cal_month - 3*(num_periods%4)
    prev_year = cal_year - int(num_periods/4)

    if prev_month == 0:
        prev_month = prev_month+12
        prev_year = prev_year - 1

    start_date = (prev_year*100+prev_month)*100 + month_day_dict[prev_month]
    end_date = (cal_year*100+cal_month)*100 + month_day_dict[cal_month]
    
    if verbose:
        print("Returning data from: {} to: {}".format(start_date, end_date))
        
    # Selecting companies which are in S&P500 on given calendar quarter (cal_qtr)
    # and have atleast num_periods data points in history; Dropping other companies
    
    t_mask = np.logical_and(data_df['data_avail_date'] > start_date, data_df['data_avail_date'] <= end_date)
    t_mask = np.logical_and(t_mask, data_df['gvkey'].isin(get_sp500_constituents(cal_qtr)))
    
    sub_df = data_df.loc[t_mask, :].copy()
    
    # Picking the 'num_periods' most recent observations
    # if num_periods=10, will select the 10 most recent observations, dropping others
    sub_df = sub_df.sort_values(['gvkey', 'data_avail_date']).groupby(['gvkey']).head(num_periods)
    
    # Dropping companies with less than 'num_periods' data points
    t_cols = ['data_avail_date', 'datadate']
    t_df = (sub_df.groupby('gvkey').count().sort_values('data_avail_date', ascending=False).loc[:,t_cols] == num_periods)
    drop_list = t_df.loc[~t_df[t_cols[0]]].index.tolist()
    
    return sub_df.loc[~sub_df['gvkey'].isin(drop_list), :]


def get_meta_data():
    col_descriptions = dict()
    col_descriptions['cshprq'] = 'Common Shares Used to Calculate Earnings per Share (Basic)'
    col_descriptions['ajexq'] = 'Adjustment Factor (Cumulative) by Ex-Date'
    col_descriptions['ibcy'] = 'Income Before Extraordinary Items (Cash Flow)'
    col_descriptions['dpcy'] = 'Depreciation and Amortization (Cash Flow)'
    col_descriptions['capxy'] = 'Capital Expenditures (Statement of Cash Flows)'
    col_descriptions['ibq'] = 'Income Before Extraordinary Items'
    col_descriptions['dpq'] = 'Depreciation and Amortization'
    col_descriptions['txtq'] = 'Income Taxes'
    col_descriptions['piq'] = 'Pretax Income'
    col_descriptions['ibcomq'] = 'Income Before Extraordinary Items – Adjusted for Common Stock Equivalents'
    col_descriptions['epsfxq'] = 'Earnings per Share (Diluted) – Excluding Extraordinary Items'
    col_descriptions['xintq'] = 'Interest Expense'
    col_descriptions['oibdpq'] = 'Operating Income Before Depreciation'
    col_descriptions['cogsq'] = 'Cost of Goods Sold'
    col_descriptions['saleq'] = 'Sales (Net)'
    col_descriptions['nopiq'] = 'Nonoperating Income (Expense)'
    col_descriptions['xrdq'] = 'Research and Development Expense'
    col_descriptions['xsgaq'] = 'Selling, General, and Administrative Expenses'
    col_descriptions['rectq'] = 'Receivables – Total'
    col_descriptions['invtq'] = 'Inventories'
    col_descriptions['atq'] = 'Total Net Assets (Assets – Total/Liabilities and Stockholders’ Equity – Total)'
    col_descriptions['dpactq'] = 'Depreciation, Depletion, and Amortization (Accumulated)'
    col_descriptions['icaptq'] = 'Invested Capital – Total'
    col_descriptions['ppentq'] = 'Property, Plant, and Equipment – Total (Net)'
    col_descriptions['actq'] = 'Current Assets – Total'
    col_descriptions['lctq'] = 'Current Liabilities – Total'
    col_descriptions['dlttq'] = 'Long- Term Debt - Total'
    col_descriptions['dlcq'] = 'Debt in Current Liabilities'
    col_descriptions['ceqq'] = 'Common Equity – Total'
    col_descriptions['reunaq'] = 'Unadjusted Retained Earnings'
    col_descriptions['acominq'] = 'Accumulated Other Comprehensive Income (Loss)'
    col_descriptions['seqoq'] = 'Other Stockholders Equity Adjustments'
    col_descriptions['cheq'] = 'Cash & Short-term Investments'

    info_columns = ['gvkey', 'datadate', 'rdq', 'fyearq', 'fqtr', 'fyr', 'tic', 'data_avail_date']
    sel_columns = ['cshprq', 'ajexq', 'ibcy', 'dpcy', 'capxy', 'ibq', 'dpq', 'txtq', 'piq', 'ibcomq',
                   'epsfxq', 'xintq', 'oibdpq', 'cogsq', 'saleq', 'nopiq', 'xrdq', 'xsgaq', 'rectq', 'invtq',
                   'atq', 'dpactq', 'icaptq', 'ppentq', 'actq', 'lctq', 'dlttq', 'dlcq', 'ceqq', 'reunaq', 'acominq', 'seqoq', 'cheq']

    return info_columns, sel_columns, col_descriptions


# ##### USING FUNCTION TO GET DATA

#t_df = get_co_data(data_df = sp500.loc[:, info_columns+sel_columns], cal_qtr=201606, num_periods=100, verbose=True)


####################################
### PORTFOLIO FUNCTIONS ############
####################################


def get_company_universe(fund_data_df, min_periods=200):
    # gets company gvkeys with atleast 200 quarters data & one-to-one mapping with CRSP data
    gvkey_df = fund_data_df.groupby('gvkey').count().loc[:, ['datadate', 'rdq']]>=min_periods
    gvkey_list = gvkey_df.loc[gvkey_df['datadate']].index.tolist()
    
    crsp_comp_link = pd.read_csv('crsp_compustat_merged.csv')
    t_group_df = crsp_comp_link.loc[crsp_comp_link['GVKEY'].isin(gvkey_list),['GVKEY', 'LPERMNO']].drop_duplicates().groupby('GVKEY').count().sort_values('LPERMNO', ascending=False)
    sel_gvkeys = t_group_df[t_group_df['LPERMNO'] == 1].index.tolist()
    
    print('# of unique companies with atleast {} quarters fundamental data: {}'.format(min_periods, len(gvkey_list)))
    print('# of companies with one-to-one mapping in CRSP-COMP link table: {}'.format(len(sel_gvkeys)))
    
    compNo = 0
    universe_gvkeys = []
    for gv in gvkey_list:
        if gv in sel_gvkeys:
            compNo = compNo+1
            universe_gvkeys.append(gv)

    print('# of companies with atleast {} quarters data & one-to-one mapping in CRSP: {}'.format(min_periods, compNo))
    
    gvkey_lpermno_dict = crsp_comp_link.loc[crsp_comp_link['GVKEY'].isin(universe_gvkeys), ['GVKEY', 'LPERMNO']].drop_duplicates().set_index('GVKEY').to_dict()['LPERMNO']
    return universe_gvkeys, gvkey_lpermno_dict


def temp_func(x):
    t_series = pd.Series()
    t_series['RET'] = np.cumprod((1+x['RET']).as_matrix())[-1] - 1
    t_series['PRC'] = x.loc[x.index[-1], 'PRC']
    return t_series


def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


def get_sp500_clean_monthly_returns(file_name='sp500_price_data.csv'):
    sp500_monthly_prices = pd.read_csv(file_name)
    sp500_monthly_prices = sp500_monthly_prices.loc[sp500_monthly_prices['PERMNO'].isin(gvkey_lpermno_dict.values()), :]
    sp500_monthly_prices = sp500_monthly_prices.loc[~sp500_monthly_prices['RET'].isin(['C', 'B']), :]
    sp500_monthly_prices['RET'] = sp500_monthly_prices['RET'].astype('float')
    sp500_monthly_prices['PRC'] = sp500_monthly_prices['PRC'].astype('float')
    sp500_monthly_prices['data_cal_qtr'] = sp500_monthly_prices['date'].apply(lambda x: pd.to_datetime(str(x)).year*100+pd.to_datetime(str(x)).quarter*3)

    sub_df = sp500_monthly_prices.loc[:, ['PERMNO', 'data_cal_qtr', 'PRC', 'RET']].sort_values(['PERMNO', 'data_cal_qtr'])
    return_data = sub_df.groupby(['PERMNO', 'data_cal_qtr']).apply(lambda x: temp_func(x)).reset_index().rename(columns={'PERMNO': 'LPERMNO', 'data_cal_qtr': 'data_cal_qtr', 0: 'RET'})

    return return_data


def get_merged_company_data(data_df, sp500_clean_returns, cal_qtr=201712, num_periods=200, verbose=True):
        
    t_df = get_co_data(data_df=data_df, cal_qtr=cal_qtr, num_periods=num_periods, verbose=verbose)
    print('# of companies selected: {}'.format(len(t_df['gvkey'].unique())))
    
    t_df['data_avail_cal_qtr'] = t_df['data_avail_date'].apply(lambda x: pd.to_datetime(str(int(x))).year*100+pd.to_datetime(str(int(x))).quarter*3)
    t_df['data_cal_qtr'] = t_df['datadate'].apply(lambda x: pd.to_datetime(str(int(x))).year*100+pd.to_datetime(str(int(x))).quarter*3)
    
    t_df['LPERMNO'] = t_df['gvkey'].apply(lambda x: gvkey_lpermno_dict[x] if x in gvkey_lpermno_dict.keys() else None)
    
    merge_df = pd.merge(t_df.loc[~pd.isnull(t_df['LPERMNO']), :], sp500_clean_returns, on=['LPERMNO', 'data_cal_qtr'], how='inner')
    merge_df = merge_df.sort_values(['gvkey', 'datadate', 'rdq'], ascending=False).groupby(['gvkey', 'datadate']).head(1)

    return merge_df


def get_sp500_daily_prices(filename='sp500_price_daily_sel_universe.csv'):
    daily_prices = pd.read_csv(filename)
    daily_prices = daily_prices.loc[~daily_prices['RET'].isin(['C', 'B']), :]

    daily_prices = daily_prices.drop_duplicates(['date', 'PERMNO'])
    daily_prices['date'] = pd.to_datetime(daily_prices['date'], format='%Y%m%d')
    
    return daily_prices


def analyze_portfolio_performance(merge_df, daily_prices_pivot_df, portf_col, num_quantiles,
                                  long_quantile, short_quantile, plot_ret=True, verbose=True):

    data_df = merge_df.loc[:, ['gvkey', portf_col, 'data_avail_cal_qtr']].copy().sort_values('data_avail_cal_qtr')

    if verbose:
        print('Selecting the latest available data point for each quarter.')
    data_df = data_df.sort_values(['gvkey', 'data_avail_cal_qtr'], ascending=False).groupby(['gvkey', 'data_avail_cal_qtr']).head(1)

    data_df['quantile'] = 0
    data_df['portf_weight'] = 0
    error_dates = []
    
    if verbose:
        print('Creating quantile portfolios...')
        
    for d in data_df['data_avail_cal_qtr'].unique():
        sub_df = data_df.loc[data_df['data_avail_cal_qtr'] == d, :]
        try:
            t_quantiles = pd.qcut(sub_df[portf_col], q=num_quantiles, labels=range(1, num_quantiles))
        except ValueError:
            error_dates.append(d)
            t_quantiles = pct_rank_qcut(sub_df[portf_col], n=num_quantiles)

        take_position = False

        #### ONLY TAKE POSITION IF ATLEAST 50% of STOCKS ARE ASSIGNED TO SOME
        #### QUANTILE; AVOID POSITIONS WHEN DATA IS SAME FOR ALL STOCKS
        if 1.0*sum(t_quantiles == 0)/sub_df.shape[0] <= 0.5:
            take_position = True
        
        #set_trace()
        if take_position:
            long_count = sum(t_quantiles.isin(long_quantile))
            short_count = sum(t_quantiles.isin(short_quantile))
            position_matrix = pd.Series(np.zeros(t_quantiles.shape[0]), index=t_quantiles.index)
            position_matrix[t_quantiles.isin(long_quantile)] = 1.0/long_count
            position_matrix[t_quantiles.isin(short_quantile)] = -1.0/short_count
            data_df.loc[data_df['data_avail_cal_qtr'] == d, 'portf_weight'] = position_matrix

        data_df.loc[data_df['data_avail_cal_qtr'] == d, 'quantile'] = t_quantiles
        
    if verbose:
        print('Finished creating quantile portfolios. Upsampling holdings to daily frequency...')
    data_df['portf_weight'] = data_df['portf_weight'].fillna(0)
    data_df['PERMNO'] = data_df['gvkey'].apply(lambda x: gvkey_lpermno_dict[x] if x in gvkey_lpermno_dict.keys() else None)
    data_df['weight_avail_date'] = data_df['data_avail_cal_qtr'].apply(lambda x: pd.to_datetime(x, format='%Y%m') + pd.offsets.MonthEnd(0))

    weights_df = data_df.pivot(index='weight_avail_date', columns='PERMNO', values='portf_weight').fillna(0).resample('D').ffill()
    
    if verbose:
        print('Merging holdings data with daily returns data...')
    weight_ret_df = pd.merge(daily_prices_pivot_df.loc[:, weights_df.columns], weights_df, 
                         left_index=True, right_index=True, how='inner', suffixes=('_ret', '_weight')).fillna(0)

    weight_ret_df = weight_ret_df.astype('float')
    return_cols = [str(c)+'_ret' for c in weights_df]
    weights_cols = [str(c)+'_weight' for c in weights_df]

    ret_matrix = np.multiply(weight_ret_df.loc[:, return_cols].as_matrix(), weight_ret_df.loc[:, weights_cols].as_matrix()).sum(axis=1)
    if verbose:
        print('Calculating portfolio returns...')
    portf_returns = pd.DataFrame(ret_matrix, index=weight_ret_df.index, columns=['portf_returns'])
    
    if plot_ret:
        fig, axes = plt.subplots(1,1, figsize=(20, 6))
        (portf_returns+1).cumprod().plot(ax=axes)
        axes.set_title('Cumulative returns of portfolio based on - ' + portf_col)
        plt.show()
    
    return portf_returns
