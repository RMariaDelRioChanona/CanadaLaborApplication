import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from statsmodels.tsa.stattools import ccf 
import scipy
import statsmodels.api as sm
import copy


def load_and_filter_data(file_path, filters, date_column, start_date):
    """
    Load data from a CSV file, apply multiple filters, convert a date column to datetime, and filter by a start date.
    
    Args:
    file_path (str): Path to the CSV file.
    filters (dict): Dictionary where keys are column names and values are the filter values required for those columns.
    date_column (str): Name of the column containing date information.
    start_date (str): Start date from which to filter the data in YYYY-MM-DD format.
    
    Returns:
    DataFrame: The processed pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    for key, value in filters.items():
        df = df[df[key] == value]
    df[date_column] = pd.to_datetime(df[date_column])
    df = df[df[date_column] >= start_date]
    return df


def calculate_growth_rate(df, value_col):
    df['Growth_Rate'] = (df[value_col] - df[value_col].shift(1)) / df[value_col].shift(1) * 100
    mean_rate = df['Growth_Rate'].mean()
    std_rate = df['Growth_Rate'].std()
    df['Normalized_Growth_Rate'] = (df['Growth_Rate'] - mean_rate) / std_rate
    return df

def calculate_fluctuations(df, value_col):
    '''
    Calculate deviations from the linear trend and normalize them for a time series column in a DataFrame.
    '''
    growth = (df[value_col].iloc[-1] - df[value_col].iloc[0]) / len(df)
    df['Fluctuations'] = df[value_col] - growth*(df.index - df.index[0])
    df['Fluctuations_norm'] = df['Fluctuations']/df['Fluctuations'].mean()
    return df

def plot_time_series(axes, index, x_data, y_data, title, ylabel):
    axes[index].plot(x_data, y_data, linewidth=3)
    axes[index].set_title(title, fontsize=16)
    axes[index].set_ylabel(ylabel, fontsize=16)
    axes[index].tick_params(axis='both', labelsize=12)
    axes[index].grid(True)

def calculate_rate_change(df, value_col):
    """
    Calculate the change in rate over time.

    Args:
    df (DataFrame): DataFrame containing the data.
    value_col (str): Name of the column containing the rate values.

    Returns:
    Series: Series containing the rate changes.
    """
    rate_change = df[value_col].diff() / df[value_col].shift(1)
    return rate_change


def filter_data_by_common_dates(df, index, common_dates):
    """
    Filter DataFrame to only include rows with common dates.

    Args:
    df (DataFrame): DataFrame containing the data.
    index (Index): Index object corresponding to the DataFrame's date column.
    common_dates (Index): Index object containing the common dates.

    Returns:
    DataFrame: Filtered DataFrame with rows corresponding to common dates.
    """
    return df[df['REF_DATE'].isin(common_dates)]

def synchronize_dataframes(*dfs, date_column='REF_DATE'):
    """
    Synchronize multiple DataFrames based on common dates.

    Args:
    *dfs (DataFrame): Variable number of DataFrames to be synchronized.
    date_column (str): Name of the date column in the DataFrames.

    Returns:
    tuple: Tuple containing synchronized DataFrames.
    """
    # Find common dates using intersection
    common_dates = set(dfs[0][date_column]).intersection(*(set(df[date_column]) for df in dfs[1:]))

    # Filter DataFrames to only include rows with common dates
    synced_dfs = tuple(df[df[date_column].isin(common_dates)] for df in dfs)

    return synced_dfs


# Define constants
path_data = "../data/CanadaData/"
path_fig = "../results/fig/"
start_date = '2006-01-01'
date_column = 'REF_DATE'


# files
file_gdp = path_data + "3610043401_GDP.csv"
file_unemp = path_data + "1410037401_employment_unemp.csv"
file_uns = path_data + "unemployment_rate_seasonal.csv"
file_vac = path_data + "1410040601_vacancies.csv"
file_bb = path_data + "Standardized_CFI_Business.csv"

#df_uns = pd.read_csv(file_uns)
df_bb = pd.read_csv(file_bb)

# filters for data
gdp_filters = {'North American Industry Classification System (NAICS)': \
               'All industries [T001]'}
unemp_filters = {
    'Labour force characteristics': 'Unemployment rate',
    'Population centre and rural areas': 'Total, all population centres and rural areas'
}
vac_filters = {
    'North American Industry Classification System (NAICS)': 'Total, all industries',
    'Statistics': 'Job vacancy rate'
}

# Load and process GDP data, unemp, vac
df_gdp = load_and_filter_data(file_gdp, gdp_filters, 'REF_DATE', start_date)
df_gdp = calculate_growth_rate(df_gdp, 'VALUE')
df_gdp = calculate_fluctuations(df_gdp, "VALUE")
df_unemp = load_and_filter_data(file_unemp, unemp_filters, 'REF_DATE', start_date)
df_uns = pd.read_csv(file_uns)
df_uns['REF_DATE'] = pd.to_datetime(df_uns['REF_DATE'])
# df_uns = df_uns[df_uns['REF_DATE'] >= '2015']
df_vac = load_and_filter_data(file_vac, vac_filters, 'REF_DATE', start_date)
# Calculate rate changes

# for gdp calculcate growth
df_gdp["Change"] = calculate_rate_change(df_gdp, 'VALUE')
df_unemp["Change"] = calculate_rate_change(df_unemp, 'VALUE')
df_uns["Change"] = calculate_rate_change(df_uns, 'VALUE')
df_vac["Change"] = calculate_rate_change(df_vac, 'VALUE')


### Get average unemployment rate and vacancy rate pre covid and from 2022 until now

print( "Unemployment 2011-2020 ",df_unemp['VALUE'][df_unemp['REF_DATE'] < '2020'].mean())
print("Unemployment 2022-2024 ",df_unemp['VALUE'][df_unemp['REF_DATE'] > '2022'].mean())
print( "Unemployment 2011-2020 ",df_uns['VALUE'][df_uns['REF_DATE'] < '2020'].mean())
print("Unemployment 2022-2024 ",df_uns['VALUE'][df_uns['REF_DATE'] > '2022'].mean())
print("Vacancy 2015-2020", df_vac['VALUE'][df_vac['REF_DATE'] < '2020'].mean())
print("Vacancy 2022-2024",df_vac['VALUE'][df_vac['REF_DATE'] > '2022'].mean())


#######
# Remove trend from business barometer
#######

df_bb_fluc = calculate_fluctuations(df_bb, 'CFI_Business')
df_bb_fluc['Date'] = pd.to_datetime(df_bb_fluc['Date'])
dict_date_bbfluc = dict(zip(df_bb_fluc['Date'], df_bb_fluc['Fluctuations_norm']))

df_bb_hp = copy.copy(df_bb)
df_bb_hp['Date'] = pd.to_datetime(df_bb_hp['Date'])
df_bb_hp.set_index('Date', inplace=True)


###
# HP filter on GDP
###

df_gdp_hp = copy.copy(df_gdp)
df_gdp_hp['BBarometer_Cycle'] = df_gdp_hp['REF_DATE'].map(dict_date_bbfluc)
df_gdp_hp.set_index('REF_DATE', inplace=True)
df_gdp_hp['BBarometer_Cycle_3MA'] = df_gdp_hp['BBarometer_Cycle'].rolling(window=3, min_periods=1).mean()


# Applying the HP Filter with lambda = 129600 for monthly data
df_gdp_hp['Log_VALUE'] = np.log(df_gdp_hp['VALUE'])
cycle, trend = sm.tsa.filters.hpfilter(df_gdp_hp['Log_VALUE'], lamb=129600)

# Adding the trend and cycle to the original DataFrame
df_gdp_hp['log_Trend12'] = trend
df_gdp_hp['log_Cycle12'] = cycle
df_gdp_hp['GDP_Trend12'] = np.exp(trend)
df_gdp_hp['GDP_Cycle12'] = np.exp(cycle)


# Scale series linearly so that they have roughly same range
range_baro3ma = df_gdp_hp["BBarometer_Cycle_3MA"].max() - df_gdp_hp["BBarometer_Cycle_3MA"].min() 
range_gdp = df_gdp_hp["GDP_Cycle12"].max() - df_gdp_hp["GDP_Cycle12"].min() 

scaling_factor = range_gdp / range_baro3ma

# Scale the original series
df_gdp_hp["BBarometer_Cycle_3MA"] = df_gdp_hp["BBarometer_Cycle_3MA"] * scaling_factor

# Adjust the mean of the transformed series to match the original mean of 1
df_gdp_hp["BBarometer_Cycle_3MA"] = df_gdp_hp["BBarometer_Cycle_3MA"] - df_gdp_hp["BBarometer_Cycle_3MA"].mean() + 1

# Scale series linearly so that they have roughly same range
range_baro = df_gdp_hp["BBarometer_Cycle"].max() - df_gdp_hp["BBarometer_Cycle"].min() 
range_gdp = df_gdp_hp["GDP_Cycle12"].max() - df_gdp_hp["GDP_Cycle12"].min() 

scaling_factor = range_gdp / range_baro

# Scale the original series
df_gdp_hp["BBarometer_Cycle"] = df_gdp_hp["BBarometer_Cycle"] * scaling_factor

# Adjust the mean of the transformed series to match the original mean of 1
df_gdp_hp["BBarometer_Cycle"] = df_gdp_hp["BBarometer_Cycle"] - df_gdp_hp["BBarometer_Cycle"].mean() + 1


df_gdp_hp['GBP_BB3MA_minmax_interact'] = df_gdp_hp.apply(
    lambda row: row['GDP_Cycle12'] if row['GDP_Cycle12'] < 1 else (row['BBarometer_Cycle_3MA'] if row['BBarometer_Cycle_3MA'] > 1 else 1),
    axis=1
)


#####################
##### Exporting all data into one csv
#####################

df_uns.set_index('REF_DATE', inplace=True)
df_vac.set_index('REF_DATE', inplace=True)

df_uns['unemployment'] = df_uns['VALUE']  # Assuming VALUE is the unemployment rate
df_vac['vacancy'] = df_vac['VALUE']  # Assuming VALUE is the vacancy rate
df_gdp_hp = df_gdp_hp[['VALUE', 'GDP_Cycle12', 'BBarometer_Cycle', 'GBP_BB3MA_minmax_interact']] 
# df_gdp_hp = df_gdp_hp[['VALUE', 'Cycle12', 'Cycle14']]  # Assuming these are the columns for GDP and cyclical component

# Rename columns for clarity
df_gdp_hp.rename(columns={'VALUE': 'gdp', 'Cycle14': 'cycle14', 'Cycle12':'cycle12'}, inplace=True)

# Merge DataFrames
# Start by merging unemployment data with GDP data since they cover the same period
df_merged = pd.merge(df_uns['unemployment'], df_gdp_hp, left_index=True, right_index=True, how='left')

# Now merge the vacancy data which starts from 2015
df_merged = pd.merge(df_merged, df_vac[['vacancy']], left_index=True, right_index=True, how='left')

# Check the result
print(df_merged.head())
print(df_merged.tail())


df_merged.to_csv(path_data + "u_v_gdp_seasonal_barometer.csv")


#### Plotting below

plt.plot(df_gdp_hp["BBarometer_Cycle"], label="bb")
plt.plot(df_gdp_hp['GDP_Cycle12'], label='gdp')
plt.plot(df_gdp_hp["BBarometer_Cycle"]**2, label='bb**2')
plt.plot(df_gdp_hp['GDP_Cycle12']**2, label='gdp**2')
plt.plot(df_gdp_hp['GBP_BB3MA_minmax_interact'], label='gdp_bb')
plt.legend()
plt.show()

#################################################################

# plt.plot(df_gdp_hp["BBarometer_Cycle"], label="bb")
# plt.plot(df_gdp_hp['GDP_Cycle12'], label='gdp')
plt.plot(df_gdp_hp["BBarometer_Cycle_3MA"]**2, label='bb**2')
# plt.plot(df_gdp_hp['GDP_Cycle12']**2, label='gdp**2')
plt.plot(df_gdp_hp['GBP_BB3MA_minmax_interact'], label='gdp_bb')
plt.legend()
plt.show()

plt.plot(df_gdp_hp["BBarometer_Cycle_3MA"], label="bb")
plt.plot(df_gdp_hp['GDP_Cycle12'], label='gdp')
plt.legend()
plt.show()

plt.plot((df_gdp_hp['BusinessBaroCycle']), label='norm barometer')
plt.axhline(y=1, color='k')
plt.legend()
plt.show()

plt.plot(df_gdp_hp['Cycle12'], label='gdp')
plt.plot(1 + 0.42*(df_gdp_hp['BusinessBaroCycle']-1), label='norm barometer')
plt.axhline(y=1, color='k')
plt.legend()
plt.savefig(path_fig + 'gdp_barometer_norm.png')
plt.show()


plt.plot(df_gdp_hp['Cycle12'], label='gdp')
plt.plot(1 + 0.42*(df_gdp_hp['BusinessBaroCycle']-1), label='norm barometer')
plt.plot(df_gdp_hp['cycle_bb_mix'], '.')
plt.axhline(y=1, color='k')
plt.legend()
plt.savefig(path_fig + 'gdp_barometer_norm_mix.png')
plt.show()

# Plotting in three panels
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 16), sharex=True)
plt.title("lambda = 129600")
# Plot Unemployment rate
plot_time_series(axes, 0, df_gdp_hp.index, df_gdp_hp['Log_VALUE'], 'Log GDP (seasonaly adjusted) lambda = 129600', 'log GDP')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp_hp.index, df_gdp_hp['log_Cycle12'], 'log GDP cycle', 'log GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_gdp_hp.index, df_gdp_hp['log_Trend12'], 'log Trend', 'log GDP')

plot_time_series(axes, 3, df_gdp_hp.index, df_gdp_hp['VALUE'], 'GDP (seasonaly adjusted)', 'GDP')
# Plot
plot_time_series(axes, 4, df_gdp_hp.index, df_gdp_hp['Cycle12'], 'Cycle', 'GDP')

plot_time_series(axes, 5, df_gdp_hp.index, df_gdp_hp['Trend12'], 'Trend', 'GDP')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "gpds_hp129.png")
plt.show()


### other value

# df_gdp_hp = copy.copy(df_gdp)
# df_gdp_hp.set_index('REF_DATE', inplace=True)
# df_gdp_hp['Log_VALUE'] = np.log(df_gdp_hp['VALUE'])

# Applying the HP Filter with lambda = 14400 for monthly data
cycle, trend = sm.tsa.filters.hpfilter(df_gdp_hp['Log_VALUE'], lamb=14400)

# Adding the trend and cycle to the original DataFrame
df_gdp_hp['log_Trend14'] = trend
df_gdp_hp['log_Cycle14'] = cycle
df_gdp_hp['Trend14'] = np.exp(trend)
df_gdp_hp['Cycle14'] = np.exp(cycle)


# Plotting in three panels
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 16), sharex=True)
plt.title("lambda = 14400")
# Plot Unemployment rate
plot_time_series(axes, 0, df_gdp_hp.index, df_gdp_hp['Log_VALUE'], 'Log GDP (seasonaly adjusted) lambda = 14400', 'log GDP')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp_hp.index, df_gdp_hp['log_Cycle14'], 'log GDP cycle', 'log GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_gdp_hp.index, df_gdp_hp['log_Trend14'], 'log Trend', 'log GDP')

plot_time_series(axes, 3, df_gdp_hp.index, df_gdp_hp['VALUE'], 'GDP (seasonaly adjusted)', 'GDP')
# Plot
plot_time_series(axes, 4, df_gdp_hp.index, df_gdp_hp['Cycle14'], 'Cycle', 'GDP')

plot_time_series(axes, 5, df_gdp_hp.index, df_gdp_hp['Trend14'], 'Trend', 'GDP')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "gpds_hp144.png")
plt.show()

#####################
##### Exporting all data into one csv
#####################

df_uns.set_index('REF_DATE', inplace=True)
df_vac.set_index('REF_DATE', inplace=True)

df_uns['unemployment'] = df_uns['VALUE']  # Assuming VALUE is the unemployment rate
df_vac['vacancy'] = df_vac['VALUE']  # Assuming VALUE is the vacancy rate
df_gdp_hp = df_gdp_hp[['VALUE', 'Cycle12', 'Cycle14', 'BB_cycle_norm', 'cycle_bb_mix']] 
# df_gdp_hp = df_gdp_hp[['VALUE', 'Cycle12', 'Cycle14']]  # Assuming these are the columns for GDP and cyclical component

# Rename columns for clarity
df_gdp_hp.rename(columns={'VALUE': 'gdp', 'Cycle14': 'cycle14', 'Cycle12':'cycle12'}, inplace=True)

# Merge DataFrames
# Start by merging unemployment data with GDP data since they cover the same period
df_merged = pd.merge(df_uns['unemployment'], df_gdp_hp, left_index=True, right_index=True, how='left')

# Now merge the vacancy data which starts from 2015
df_merged = pd.merge(df_merged, df_vac[['vacancy']], left_index=True, right_index=True, how='left')

# Check the result
print(df_merged.head())
print(df_merged.tail())


plt.plot(df_merged['unemployment'][df_merged.index > '2015'], df_merged['vacancy'][df_merged.index > '2015'], "o-")
plt.show()


df_merged.to_csv(path_data + "u_v_gdp_seasonal_barometer.csv")


# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
# Plot Unemployment rate
plot_time_series(axes, 0, df_uns['REF_DATE'], df_uns['VALUE'], 'Unemployment Rate (seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp_hp.index, df_gdp_hp['Cycle'], 'GDP (seasonally adjusted)', 'GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "us_gpdshp_vs.png")
plt.show()



# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
# Plot Unemployment rate
plot_time_series(axes, 0, df_uns['REF_DATE'], df_uns['VALUE'], 'Unemployment Rate (seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['VALUE'], 'GDP (seasonally adjusted)', 'GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "us_gpds_vs.png")
plt.show()


# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
# Plot Unemployment rate
plot_time_series(axes, 0, df_uns['REF_DATE'], df_uns['VALUE'], 'Unemployment Rate (seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['VALUE'], 'GDP (seasonally adjusted)', 'GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "us_gpds_vs.png")
plt.show()


# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot Unemployment rate
plot_time_series(axes, 0, df_uns['REF_DATE'], df_uns['VALUE'], 'Unemployment Rate (seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['Fluctuations_norm'], 'GDP growth normalized and standardized', 'Fluctuations')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(path_fig + "us_gpds_norm_vs.png")
plt.show()

# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot Unemployment rate
plot_time_series(axes, 0, df_unemp['REF_DATE'], df_unemp['VALUE'], 'Unemployment Rate (not seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['VALUE'], 'GDP', 'GDP')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.show()



# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot Unemployment rate
plot_time_series(axes, 0, df_unemp['REF_DATE'], df_unemp['VALUE'], 'Unemployment Rate (not seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['Fluctuations'], 'GDP', 'Fluctuations')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.show()


# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot Unemployment rate
plot_time_series(axes, 0, df_unemp['REF_DATE'], df_unemp['VALUE'], 'Unemployment Rate (not seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['Fluctuations_norm'], 'GDP', 'Fluctuations')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.show()

# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Plot Unemployment rate
plot_time_series(axes, 0, df_unemp['REF_DATE'], df_unemp['VALUE'], 'Unemployment Rate (not seasonaly adjusted)', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['Normalized_Growth_Rate'], 'Normalized GDP Growth Rate', 'Normalized Rate')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['VALUE'], 'Job Vacancy Rate (seasonaly adjusted)', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.show()

# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
# Plot Unemployment rate
plot_time_series(axes, 0, df_uns['REF_DATE'], df_uns['Change'], 'Unemployment Rate Change', 'Rate (%)')
# Plot Normalized GDP Growth Rate
plot_time_series(axes, 1, df_gdp['REF_DATE'], df_gdp['Change'], 'GDP Change', 'Normalized Rate')
# Plot Job Vacancy Rate
plot_time_series(axes, 2, df_vac['REF_DATE'], df_vac['Change'], 'Job Vacancy Rate Change', 'Rate (%)')
# Set date formatter and locator for all axes
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year
axes[0].set_ylim([-0.2, 0.3])
axes[1].set_ylim([-0.015, 0.015])
axes[1].set_ylim([-0.015, 0.015])
for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.show()



####
# Cross correlation analysis
####

# Synchronize dataframes
df_gdp_sync, df_unemp_sync, df_vac_sync = synchronize_dataframes(df_gdp, df_unemp, df_vac)
df_gdp_sync, df_uns_sync, df_vac_sync = synchronize_dataframes(df_gdp, df_uns, df_vac)

# Compute rate changes for unemployment and vacancy
unemp_rate_change = calculate_rate_change(df_unemp_sync, 'VALUE').iloc[1:]
unemp_rate_change = calculate_rate_change(df_uns_sync, 'VALUE').iloc[1:]
vac_rate_change = calculate_rate_change(df_vac_sync, 'VALUE').iloc[1:]
gdp_change = calculate_rate_change(df_gdp_sync, 'VALUE').iloc[1:]
gdp_norm_growth_rate = df_gdp_sync['Normalized_Growth_Rate'].iloc[1:]


len(unemp_rate_change)
len(gdp_norm_growth_rate)

plt.plot(unemp_rate_change, gdp_change, "o")
# plt.ylim([-0.015, 0.02])
# plt.xlim([-0.015, 0.02])
plt.title("Okun's law for Canada ", fontsize=14)
plt.xlabel('Change in unemployment rate', fontsize=14)
plt.ylabel('Change in GDP', fontsize=14)
plt.tight_layout()
plt.savefig(path_fig + "canada_okuns_law.png")
plt.show()

plt.plot(unemp_rate_change, gdp_change, "o")
plt.ylim([-0.015, 0.02])
plt.xlim([-0.15, 0.2])
plt.title("Okun's law for Canada (excluding outliers)", fontsize=14)
plt.xlabel('Change in unemployment rate', fontsize=14)
plt.ylabel('Change in GDP', fontsize=14)
plt.tight_layout()
plt.savefig(path_fig + "canada_okuns_law_noout.png")
plt.show()

plt.plot(vac_rate_change, gdp_change, "o")
plt.ylim([-0.015, 0.02])
plt.title("Vacancy rate vs GDP (excluding outliers)", fontsize=14)
plt.xlabel('Change in vacancy rate', fontsize=14)
plt.ylabel('Change in GDP', fontsize=14)
plt.tight_layout()
plt.savefig(path_fig + "vac_rate_gdp_noout.png")
plt.show()



# Cross-correlation analysis
lags = np.arange(-20, 20)  # Adjust the range of lags based on your specific needs
ccf_gdp_unemp = [ccf(gdp_norm_growth_rate, unemp_rate_change, adjusted=False)[lag] for lag in lags]
ccf_gdp_vac = [ccf(gdp_norm_growth_rate, vac_rate_change, adjusted=False)[lag] for lag in lags]

# Plotting the cross-correlation
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.stem(lags, ccf_gdp_unemp, use_line_collection=True)
plt.title('Cross-Correlation between Normalized GDP Growth Rate and Unemployment Rate Change')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')

plt.subplot(1, 2, 2)
plt.stem(lags, ccf_gdp_vac, use_line_collection=True)
plt.title('Cross-Correlation between Normalized GDP Growth Rate and Vacancy Rate Change')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')

plt.tight_layout()
plt.show()


################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from statsmodels.tsa.stattools import ccf 

# Load data from provided files
path_data = "../data/CanadaData/"
file_gdp = "3610043401_GDP.csv"
file_vac = "1410040601_vacancies.csv"
file_unemp = "1410037401_employment_unemp.csv"

df_unemp = pd.read_csv(path_data + file_unemp)
df_vac = pd.read_csv(path_data + file_vac)
df_gdp = pd.read_csv(path_data + file_gdp)

# Selecting nationwide data
df_gdp = df_gdp[df_gdp['North American Industry Classification System (NAICS)'] == 'All industries [T001]']
df_unemp = df_unemp[(df_unemp['Labour force characteristics'] == 'Unemployment rate') &
                    (df_unemp['Population centre and rural areas'] == 'Total, all population centres and rural areas')]
df_vac = df_vac[(df_vac['North American Industry Classification System (NAICS)'] == 'Total, all industries') &
                (df_vac['Statistics'] == 'Job vacancy rate')]

# Convert REF_DATE to datetime
df_gdp['REF_DATE'] = pd.to_datetime(df_gdp['REF_DATE'])
df_unemp['REF_DATE'] = pd.to_datetime(df_unemp['REF_DATE'])
df_vac['REF_DATE'] = pd.to_datetime(df_vac['REF_DATE'])

# Filter data from January 2015 onwards
start_date = '2015-01-01'
df_gdp = df_gdp[df_gdp['REF_DATE'] >= start_date]
df_unemp = df_unemp[df_unemp['REF_DATE'] >= start_date]
df_vac = df_vac[df_vac['REF_DATE'] >= start_date]

# Calculate GDP growth rates and normalize
df_gdp['GDP_Growth_Rate'] = (df_gdp['VALUE'] - df_gdp['VALUE'].shift(1)) / df_gdp['VALUE'].shift(1) * 100
mean_growth_rate = df_gdp['GDP_Growth_Rate'].mean()
std_dev_growth_rate = df_gdp['GDP_Growth_Rate'].std()
df_gdp['Normalized_GDP_Growth_Rate'] = (df_gdp['GDP_Growth_Rate'] - mean_growth_rate) / std_dev_growth_rate

# Plotting in three panels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)

# Unemployment rate plot
axes[0].plot(df_unemp['REF_DATE'].values, df_unemp['VALUE'].values)
axes[0].set_title('Unemployment Rate')
axes[0].set_ylabel('Rate (%)')

# Normalized GDP Growth Rate plot
axes[1].plot(df_gdp['REF_DATE'].values, df_gdp['Normalized_GDP_Growth_Rate'].values)
axes[1].set_title('Normalized GDP Growth Rate')
axes[1].set_ylabel('Normalized Rate')

# Job Vacancy Rate plot
axes[2].plot(df_vac['REF_DATE'].values, df_vac['VALUE'].values)
axes[2].set_title('Job Vacancy Rate')
axes[2].set_ylabel('Rate (%)')
axes[2].set_xlabel('Date')

# Setting the date formatter and locator
date_formatter = DateFormatter('%Y')  # Formats the date as 'Year'
year_locator = YearLocator()  # Locates one tick per year

for ax in axes:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(date_formatter)
    ax.grid(True)  # Optional: Adds a grid for easier readability

plt.tight_layout()
plt.show()


# Convert date columns to Index
index_gdp = pd.Index(df_gdp['REF_DATE'])
index_unemp = pd.Index(df_unemp['REF_DATE'])
index_vac = pd.Index(df_vac['REF_DATE'])

# Find common dates using intersection
common_dates = index_gdp.intersection(index_unemp).intersection(index_vac)

# Filter DataFrames to only include rows with common dates
df_gdp_sync = df_gdp[df_gdp['REF_DATE'].isin(common_dates)]
df_unemp_sync = df_unemp[df_unemp['REF_DATE'].isin(common_dates)]
df_vac_sync = df_vac[df_vac['REF_DATE'].isin(common_dates)]

# Compute normalized series
gdp_normalized = (df_gdp_sync['GDP_Growth_Rate'] - df_gdp_sync['GDP_Growth_Rate'].mean()) / df_gdp_sync['GDP_Growth_Rate'].std()
unemp_normalized = (df_unemp_sync['VALUE'] - df_unemp_sync['VALUE'].mean()) / df_unemp_sync['VALUE'].std()
vac_normalized = (df_vac_sync['VALUE'] - df_vac_sync['VALUE'].mean()) / df_vac_sync['VALUE'].std()

# Cross-correlation analysis
lags = np.arange(-24, 25)  # Adjust the range of lags based on your specific needs
ccf_gdp_unemp = [ccf(gdp_normalized, unemp_normalized, adjusted=False)[lag] for lag in lags]
ccf_gdp_vac = [ccf(gdp_normalized, vac_normalized, adjusted=False)[lag] for lag in lags]

# Plotting the cross-correlation
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.stem(lags, ccf_gdp_unemp, use_line_collection=True)
plt.title('Cross-Correlation between GDP Growth Rate and Unemployment Rate')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')

plt.subplot(1, 2, 2)
plt.stem(lags, ccf_gdp_vac, use_line_collection=True)
plt.title('Cross-Correlation between GDP Growth Rate and Vacancy Rate')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')

plt.tight_layout()
plt.show()