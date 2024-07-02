import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


years= [2021, 2025, 2030, 2035, 2040, 2045, 2050]

# Helper function to generate a monthly range
def generate_monthly_range(start_year, end_year):
    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

def distribute_jobs(df, years=years, bulk_years=2):
    data = []
    national_data = []

    cumulative_jobs = {
        'CPY_negative': 0,
        'Op_positive': 0,
        'Op_negative': 0
    }

    for _, row in df.iterrows():
        for period_start in years:
            if row['region'] == 'Alberta.a':
                print(row['variable'],period_start)
            if period_start == 2021:
                monthly_range = generate_monthly_range(period_start, period_start + 4)
            else:
                monthly_range = generate_monthly_range(period_start, period_start + 5)
            months_in_period = len(monthly_range)
            bulk_months = bulk_years * 12
            
            # Specific job columns and their types to be computed
            job_columns = [
                ('CPY (Direct)_linear_positive', 'CPY_positive_bulk'),
                ('CPY (Direct)_linear_negative', 'CPY_negative_accumulated'),
                ('Op FTE (Direct)_linear_positive', 'Op_positive_accumulated'),
                ('Op FTE (Direct)_linear_negative', 'Op_negative_accumulated')
            ]
            
            for job_column, job_type in job_columns:
                job_value = row[job_column]
                
                if job_type == 'CPY_positive_bulk':
                    # Bulk jobs
                    jobs_per_month_bulk = job_value / bulk_months
                    for i, date in enumerate(monthly_range):
                        jobs_bulk = jobs_per_month_bulk if (date.year - period_start) < bulk_years else 0
                        bulk_entry = {
                            'region': row['region'],
                            'variable': row['variable'],
                            'model': row['model'],
                            'scenario': row['scenario'],
                            'time': date,
                            'unit': row['unit'],
                            'jobs': jobs_bulk,
                            'job_type': job_type
                        }
                        data.append(bulk_entry)
                        national_data.append(bulk_entry.copy())
                else:
                    # Accumulated jobs
                    job_per_month_acc = job_value / months_in_period
                    cumulative_key = job_type.split('_')[0] + '_' + job_type.split('_')[1]
                    cumulative_jobs_acc = cumulative_jobs[cumulative_key] + np.cumsum([job_per_month_acc] * months_in_period)
                    cumulative_jobs[cumulative_key] = cumulative_jobs_acc[-1]
                    
                    for i, date in enumerate(monthly_range):
                        acc_entry = {
                            'region': row['region'],
                            'variable': row['variable'],
                            'model': row['model'],
                            'scenario': row['scenario'],
                            'time': date,
                            'unit': row['unit'],
                            'jobs': cumulative_jobs_acc[i],
                            'job_type': job_type
                        }
                        data.append(acc_entry)
                        national_data.append(acc_entry.copy())

    # Create dataframes
    df_expanded = pd.DataFrame(data)
    
    # Aggregate national data
    national_grouped = df_expanded.groupby(['variable', 'model', 'scenario', 'time', 'unit', 'job_type'], as_index=False).sum()
    national_grouped['region'] = 'National'
    
    # Combine regional and national dataframes
    combined_df = pd.concat([df_expanded, national_grouped], ignore_index=True)

    return combined_df

# Load the data
path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"
df = pd.read_csv(path_data + file_zach)

# Filter the relevant columns
df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
         'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

# Split job creation and destruction
df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)


# NOTE this aggregation is important, since the original file has repeated entries. Discuss with Zach
# Aggregate rows with the same region, variable, and time
df = df.groupby(['region', 'variable', 'model', 'scenario', 'time', 'unit'], as_index=False).sum()



df[df['region'] == 'Alberta.a']

# Apply the function to get the combined dataframe
df_combined = distribute_jobs(df)

df_combined.to_csv(path_data + "jobs_created_destroyed_ts_general.csv", index=False)

print(len(df_combined))

jobs_types_final = ["CPY_positive_bulk", 'CPY_negative_accumulated', 'Op_positive_accumulated', 'Op_negative_accumulated']

df_jobs = df_combined[df_combined["job_type"].isin(jobs_types_final)]

df_jobs[df_jobs['time'] == '2025-01-01']

df_jobs.to_csv(path_data + "jobs_created_destroyed_ts.csv", index=False)

region_tech_data = df_jobs[(df_jobs['region'] == 'Alberta.a') & (df_jobs['variable'] == 'Electricity|Biomass|w/o CCS')]
job_data = region_tech_data[region_tech_data['job_type'] == 'Op_positive_accumulated']

job_data[job_data['time'] == '2025-01-01']

plt.plot(job_data['time'], job_data['jobs'])
print(len(job_data['time'].unique()))
print(len(job_data['time']))






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Helper function to generate a monthly range
# def generate_monthly_range(start_year, end_year):
#     return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

# def distribute_jobs(df, start_year=2021, end_year=2050, bulk_years=2):
#     data = []
#     national_data = []

#     cumulative_jobs = {
#         'CPY_positive': 0,
#         'CPY_negative': 0,
#         'Op_positive': 0,
#         'Op_negative': 0
#     }

#     for _, row in df.iterrows():
#         for period_start in range(start_year, end_year, 5):
#             monthly_range = generate_monthly_range(period_start, period_start + 4)
#             months_in_period = len(monthly_range)
#             bulk_months = bulk_years * 12
            
#             # Prepare data for each job type and assumption
#             job_columns = [
#                 ('CPY (Direct)_linear_positive', 'CPY_positive'),
#                 ('CPY (Direct)_linear_negative', 'CPY_negative'),
#                 ('Op FTE (Direct)_linear_positive', 'Op_positive'),
#                 ('Op FTE (Direct)_linear_negative', 'Op_negative')
#             ]
            
#             for job_column, job_type in job_columns:
#                 job_value = row[job_column]
                
#                 # Accumulated jobs
#                 job_per_month_acc = job_value / months_in_period
#                 cumulative_jobs_acc = cumulative_jobs[job_type] + np.cumsum([job_per_month_acc] * months_in_period)
#                 cumulative_jobs[job_type] = cumulative_jobs_acc[-1]
                
#                 # Bulk jobs
#                 jobs_per_month_bulk = job_value / bulk_months

#                 for i, date in enumerate(monthly_range):
#                     # Accumulated jobs entry
#                     acc_entry = {
#                         'region': row['region'],
#                         'variable': row['variable'],
#                         'model': row['model'],
#                         'scenario': row['scenario'],
#                         'time': date,
#                         'unit': row['unit'],
#                         'jobs': cumulative_jobs_acc[i],
#                         'job_type': f'{job_type}_accumulated'
#                     }
#                     data.append(acc_entry)

#                     # Add to national aggregation
#                     national_data.append(acc_entry.copy())
                    
#                     # Bulk jobs entry
#                     jobs_bulk = jobs_per_month_bulk if (date.year - period_start) < bulk_years else 0
#                     bulk_entry = {
#                         'region': row['region'],
#                         'variable': row['variable'],
#                         'model': row['model'],
#                         'scenario': row['scenario'],
#                         'time': date,
#                         'unit': row['unit'],
#                         'jobs': jobs_bulk,
#                         'job_type': f'{job_type}_bulk'
#                     }
#                     data.append(bulk_entry)
                    
#                     # Add to national aggregation
#                     national_data.append(bulk_entry.copy())

#     # Create dataframes
#     df_expanded = pd.DataFrame(data)
#     national_df = pd.DataFrame(national_data)
    
#     # Aggregate national data
#     national_grouped = national_df.groupby(['variable', 'model', 'scenario', 'time', 'unit', 'job_type'], as_index=False).sum()
#     national_grouped['region'] = 'National'
    
#     # Combine regional and national dataframes
#     combined_df = pd.concat([df_expanded, national_grouped], ignore_index=True)

#     return combined_df

# # Load the data
# path_data = "../data/copper/"
# file_zach = "net_new_cap_w_jobs.csv"
# df = pd.read_csv(path_data + file_zach)

# # Filter the relevant columns
# df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
#          'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

# # Split job creation and destruction
# df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
# df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
# df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
# df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

# # Apply the function to get the combined dataframe
# df_combined = distribute_jobs(df)

# # Filter the combined dataframe for the specified job types
# jobs_types_final = ["CPY_positive_bulk", 'CPY_negative_accumulated', 'Op_positive_accumulated', 'Op_negative_accumulated']
# df_jobs = df_combined[df_combined["job_type"].isin(jobs_types_final)]

# # Function to plot the time series of jobs
# def plot_jobs_time_series(df, region, technology):
#     """Plot the time series of jobs created and destroyed for a specific region and technology."""
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

#     job_types_titles = {
#         "CPY_positive_bulk": "CPY Jobs Created (Bulk)",
#         "CPY_negative_accumulated": "CPY Jobs Destroyed (Accumulated)",
#         "Op_positive_accumulated": "Op Jobs Created (Accumulated)",
#         "Op_negative_accumulated": "Op Jobs Destroyed (Accumulated)"
#     }
    
#     # Filter data for the specific region and technology
#     region_tech_data = df[(df['region'] == region) & (df['variable'] == technology)]

#     # Plot each job type in the corresponding subplot
#     for ax, job_type in zip(axs.flatten(), jobs_types_final):
#         job_data = region_tech_data[region_tech_data['job_type'] == job_type]
#         ax.plot(job_data['time'], job_data['jobs'], label=job_type)
#         ax.set_title(job_types_titles[job_type])
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Number of Jobs')
#         ax.legend()
    
#     plt.tight_layout()
#     plt.show()

# # Example usage
# plot_jobs_time_series(df_jobs, region='Alberta.a', technology='Electricity|Biomass|w/o CCS')

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function to generate a monthly range
def generate_monthly_range(start_year, end_year):
    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

def distribute_jobs(df, start_year=2021, end_year=2050, bulk_years=2):
    data = []
    national_data = []

    cumulative_jobs = {
        'CPY_positive': 0,
        'CPY_negative': 0,
        'Op_positive': 0,
        'Op_negative': 0
    }

    for _, row in df.iterrows():
        for period_start in range(start_year, end_year, 5):
            monthly_range = generate_monthly_range(period_start, period_start + 4)
            months_in_period = len(monthly_range)
            bulk_months = bulk_years * 12
            
            # Prepare data for each job type and assumption
            job_columns = [
                ('CPY (Direct)_linear_positive', 'CPY_positive'),
                ('CPY (Direct)_linear_negative', 'CPY_negative'),
                ('Op FTE (Direct)_linear_positive', 'Op_positive'),
                ('Op FTE (Direct)_linear_negative', 'Op_negative')
            ]
            
            for job_column, job_type in job_columns:
                job_value = row[job_column]
                
                # Accumulated jobs
                job_per_month_acc = job_value / months_in_period
                cumulative_jobs_acc = cumulative_jobs[job_type] + np.cumsum([job_per_month_acc] * months_in_period)
                cumulative_jobs[job_type] = cumulative_jobs_acc[-1]
                
                # Bulk jobs
                jobs_per_month_bulk = job_value / bulk_months

                for i, date in enumerate(monthly_range):
                    # Accumulated jobs entry
                    acc_entry = {
                        'region': row['region'],
                        'variable': row['variable'],
                        'model': row['model'],
                        'scenario': row['scenario'],
                        'time': date,
                        'unit': row['unit'],
                        'jobs': cumulative_jobs_acc[i],
                        'job_type': f'{job_type}_accumulated'
                    }
                    data.append(acc_entry)
                    
                    # Bulk jobs entry
                    jobs_bulk = jobs_per_month_bulk if (date.year - period_start) < bulk_years else 0
                    bulk_entry = {
                        'region': row['region'],
                        'variable': row['variable'],
                        'model': row['model'],
                        'scenario': row['scenario'],
                        'time': date,
                        'unit': row['unit'],
                        'jobs': jobs_bulk,
                        'job_type': f'{job_type}_bulk'
                    }
                    data.append(bulk_entry)
                    
    # Create dataframes
    df_expanded = pd.DataFrame(data)
    
    # Aggregate national data
    national_grouped = df_expanded.groupby(['variable', 'model', 'scenario', 'time', 'unit', 'job_type'], as_index=False).sum()
    national_grouped['region'] = 'National'
    
    # Combine regional and national dataframes
    combined_df = pd.concat([df_expanded, national_grouped], ignore_index=True)

    return combined_df

# Load the data
path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"
df = pd.read_csv(path_data + file_zach)

# Filter the relevant columns
df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
         'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

# Split job creation and destruction
df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

# Apply the function to get the combined dataframe
df_combined = distribute_jobs(df)

df_combined.to_csv(path_data + "jobs_created_destroyed_ts_general.csv", index=False)

jobs_types_final = ["CPY_positive_bulk", 'CPY_negative_accumulated', 'Op_positive_accumulated', 'Op_negative_accumulated']

df_jobs = df_combined[df_combined["job_type"].isin(jobs_types_final)]

df_jobs.to_csv(path_data + "jobs_created_destroyed_ts.csv", index=False)

region_tech_data = df_jobs[(df_jobs['region'] == 'Alberta.a') & (df_jobs['variable'] == 'Electricity|Biomass|w/o CCS')]
job_data = region_tech_data[region_tech_data['job_type'] == 'Op_positive_accumulated']
plt.plot(job_data['time'], job_data['jobs'])
print(len(job_data['time'].unique()))
print(len(job_data['time']))


###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function to generate a monthly range
def generate_monthly_range(start_year, end_year):
    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')
def distribute_jobs(df, start_year=2021, end_year=2050, bulk_years=2):
    df_expanded = []
    national_aggregated = []

    cumulative_jobs = {
        'CPY_positive': 0,
        'CPY_negative': 0,
        'Op_positive': 0,
        'Op_negative': 0
    }

    for _, row in df.iterrows():
        for period_start in range(start_year, end_year, 5):
            monthly_range = generate_monthly_range(period_start, period_start + 4)
            months_in_period = len(monthly_range)
            bulk_months = bulk_years * 12
            
            # Prepare data for each job type and assumption
            job_columns = [
                ('CPY (Direct)_linear_positive', 'CPY_positive'),
                ('CPY (Direct)_linear_negative', 'CPY_negative'),
                ('Op FTE (Direct)_linear_positive', 'Op_positive'),
                ('Op FTE (Direct)_linear_negative', 'Op_negative')
            ]
            
            for job_column, job_type in job_columns:
                job_value = row[job_column]
                
                # Accumulated jobs
                job_per_month_acc = job_value / months_in_period
                cumulative_jobs_acc = cumulative_jobs[job_type] + np.cumsum([job_per_month_acc] * months_in_period)
                cumulative_jobs[job_type] = cumulative_jobs_acc[-1]
                
                # Bulk jobs
                jobs_per_month_bulk = job_value / bulk_months

                for i, date in enumerate(monthly_range):
                    # Accumulated jobs entry
                    acc_entry = {
                        'region': row['region'],
                        'variable': row['variable'],
                        'model': row['model'],
                        'scenario': row['scenario'],
                        'time': date,
                        'unit': row['unit'],
                        'jobs': cumulative_jobs_acc[i],
                        'job_type': f'{job_type}_accumulated'
                    }
                    df_expanded.append(acc_entry)

                    # Add to national aggregation
                    national_aggregated.append(acc_entry.copy())
                    
                    # Bulk jobs entry
                    jobs_bulk = jobs_per_month_bulk if (date.year - period_start) < bulk_years else 0
                    bulk_entry = {
                        'region': row['region'],
                        'variable': row['variable'],
                        'model': row['model'],
                        'scenario': row['scenario'],
                        'time': date,
                        'unit': row['unit'],
                        'jobs': jobs_bulk,
                        'job_type': f'{job_type}_bulk'
                    }
                    df_expanded.append(bulk_entry)
                    
                    # Add to national aggregation
                    national_aggregated.append(bulk_entry.copy())

    # Create national aggregated dataframe
    national_df = pd.DataFrame(national_aggregated)
    national_grouped = national_df.groupby(['variable', 'model', 'scenario', 'time', 'unit', 'job_type'], as_index=False).sum()
    national_grouped['region'] = 'National'
    
    # Combine regional and national dataframes
    combined_df = pd.concat([pd.DataFrame(df_expanded), national_grouped], ignore_index=True)

    return combined_df


def plot_jobs_time_series(df, region, technology):
    """Plot the time series of jobs created and destroyed for a specific region and technology."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    job_types_titles = {
        "CPY_positive_bulk": "CPY Jobs Created (Bulk)",
        "CPY_negative_accumulated": "CPY Jobs Destroyed (Accumulated)",
        "Op_positive_accumulated": "Op Jobs Created (Accumulated)",
        "Op_negative_accumulated": "Op Jobs Destroyed (Accumulated)"
    }
    
    # Filter data for the specific region and technology
    region_tech_data = df[(df['region'] == region) & (df['variable'] == technology)]

    # Plot each job type in the corresponding subplot
    for ax, job_type in zip(axs.flatten(), jobs_types_final):
        job_data = region_tech_data[region_tech_data['job_type'] == job_type]
        ax.plot(job_data['time'], job_data['jobs'], label=job_type)
        ax.set_title(job_types_titles[job_type])
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Jobs')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Load the data
path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"
df = pd.read_csv(path_data + file_zach)

# Filter the relevant columns
df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
         'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

# Split job creation and destruction
df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

# Apply the function to get the combined dataframe
df_combined = distribute_jobs(df)

df_combined.to_csv(path_data + "jobs_created_destroyed_ts_general.csv", index=False)

jobs_types_final = ["CPY_positive_bulk", 'CPY_negative_accumulated', 'Op_positive_accumulated', 'Op_negative_accumulated']

df_jobs = df_combined[df_combined["job_type"].isin(jobs_types_final)]

df_jobs.to_csv(path_data + "jobs_created_destroyed_ts.csv", index=False)



region_tech_data = df_jobs[(df_jobs['region'] == 'Alberta.a') & (df_jobs['variable'] == 'Electricity|Biomass|w/o CCS')]
job_data = region_tech_data[region_tech_data['job_type'] == 'Op_positive_accumulated']
plt.plot(job_data['time'], job_data['jobs'])
len(job_data['time'].unique())
len(job_data['time'])

job_data['time'].unique()


df_jobs['variable'].unique()
n_tech = len(df_jobs['variable'].unique())

plot_jobs_time_series(df_jobs, region='Alberta.a', technology='Electricity|Biomass|w/o CCS')


# Print the first few rows of the combined results
print("Combined Jobs:")
print(df_combined.head())

















import pandas as pd
import numpy as np

# Helper function to generate a monthly range
def generate_monthly_range(start_year, end_year):
    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

# Function to distribute jobs proportionally over months and accumulate them
def jobs_accumulated(df, job_column, start_year=2021, end_year=2050):
    df_expanded = []
    for _, row in df.iterrows():
        for period_start in range(start_year, end_year, 5):
            monthly_range = generate_monthly_range(period_start, period_start + 4)
            months_in_period = len(monthly_range)
            job_per_month = row[job_column] / months_in_period
            cumulative_jobs = np.cumsum([job_per_month] * months_in_period)

            # Create a DataFrame for this specific row and add it to the list
            for i, date in enumerate(monthly_range):
                df_expanded.append({
                    'region': row['region'],
                    'variable': row['variable'],
                    'model': row['model'],
                    'scenario': row['scenario'],
                    'time': date,
                    'unit': row['unit'],
                    'jobs': cumulative_jobs[i]
                })

    return pd.DataFrame(df_expanded)

# Function to distribute jobs in bulk at the beginning of the period
def jobs_bulk(df, job_column, start_year=2021, end_year=2050, bulk_years=2):
    df_expanded = []
    for _, row in df.iterrows():
        for period_start in range(start_year, end_year, 5):
            monthly_range = generate_monthly_range(period_start, period_start + 4)
            bulk_months = bulk_years * 12
            jobs_per_month_bulk = row[job_column] / bulk_months

            # Create a DataFrame for this specific row and add it to the list
            for date in monthly_range:
                if (date.year - period_start) < bulk_years:
                    jobs = jobs_per_month_bulk
                else:
                    jobs = 0

                df_expanded.append({
                    'region': row['region'],
                    'variable': row['variable'],
                    'model': row['model'],
                    'scenario': row['scenario'],
                    'time': date,
                    'unit': row['unit'],
                    'jobs': jobs
                })

    return pd.DataFrame(df_expanded)

# Load the data
path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"
df = pd.read_csv(path_data + file_zach)

# Filter the relevant columns
df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
         'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

# Split job creation and destruction
df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

# Applying the functions
df_accumulated_cpy_positive = jobs_accumulated(df, 'CPY (Direct)_linear_positive')
df_accumulated_cpy_negative = jobs_accumulated(df, 'CPY (Direct)_linear_negative')
df_accumulated_op_positive = jobs_accumulated(df, 'Op FTE (Direct)_linear_positive')
df_accumulated_op_negative = jobs_accumulated(df, 'Op FTE (Direct)_linear_negative')

df_bulk_cpy_positive = jobs_bulk(df, 'CPY (Direct)_linear_positive')
df_bulk_cpy_negative = jobs_bulk(df, 'CPY (Direct)_linear_negative')
df_bulk_op_positive = jobs_bulk(df, 'Op FTE (Direct)_linear_positive')
df_bulk_op_negative = jobs_bulk(df, 'Op FTE (Direct)_linear_negative')


df_accumulated_cpy_negative

# Print the first few rows of the results
print("Accumulated CPY Positive Jobs:")
print(df_accumulated_cpy_positive.head())
print("Accumulated CPY Negative Jobs:")
print(df_accumulated_cpy_negative.head())
print("Accumulated Op Positive Jobs:")
print(df_accumulated_op_positive.head())
print("Accumulated Op Negative Jobs:")
print(df_accumulated_op_negative.head())

print("Bulk CPY Positive Jobs:")
print(df_bulk_cpy_positive.head())
print("Bulk CPY Negative Jobs:")
print(df_bulk_cpy_negative.head())
print("Bulk Op Positive Jobs:")
print(df_bulk_op_positive.head())
print("Bulk Op Negative Jobs:")
print(df_bulk_op_negative.head())







import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

# from matplotlib import pylab as plt
import labor_abm as lbm
import utils as ut

from matplotlib import pylab as plt

path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"

df = pd.read_csv(path_data + file_zach)
df.columns

# Filter only by columns I care

df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
       'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]

df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)


df.head()

"British Columbia.a"

years= [2021, 2025, 2030, 2035, 2040, 2045, 2050]
months_in_period = 5 * 12  # 5 years * 12 months
unique_regions = df['region'].unique()

# Split the jobs created and destroyed into separate columns



def make_accumulate_jobs(df, building_time=12):
    # Define the time periods
    years = [2021, 2025, 2030, 2035, 2040, 2045, 2050]
    months_in_period = 5 * 12  # 5 years * 12 months

    # Create a DataFrame to hold the accumulated jobs data
    accumulated_jobs = pd.DataFrame()

    # Create dictionaries to keep track of cumulative jobs
    cumulative_op_positive = {}
    cumulative_op_negative = {}
    cumulative_cpy_negative = {}

    # Loop through each row in the original DataFrame
    for index, row in df.iterrows():
        start_year = row['time']
        end_year = start_year + 4  # Each period is 5 years

        # Generate date range for the period (start of each month)
        date_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-01', freq='MS')

        # Create a temporary DataFrame for the current row
        temp_df = pd.DataFrame({
            'region': row['region'],
            'variable': row['variable'],
            'model': row['model'],
            'scenario': row['scenario'],
            'time': date_range
        })

        # Initialize job columns with zeros
        temp_df['CPY (Direct)_linear_positive'] = 0
        temp_df['CPY (Direct)_linear_negative'] = 0
        temp_df['Op FTE (Direct)_linear_positive'] = 0
        temp_df['Op FTE (Direct)_linear_negative'] = 0

        # Distribute CPY (Direct)_linear_positive jobs
        if row['CPY (Direct)_linear_positive'] > 0:
            temp_df.loc[:building_time-1, 'CPY (Direct)_linear_positive'] = row['CPY (Direct)_linear_positive'] / building_time

        # Distribute Op FTE (Direct)_linear_positive jobs (accumulate across periods)
        if row['region'] not in cumulative_op_positive:
            cumulative_op_positive[row['region']] = 0
        cumulative_op_positive[row['region']] += row['Op FTE (Direct)_linear_positive']
        if cumulative_op_positive[row['region']] > 0:
            temp_df.loc[building_time:, 'Op FTE (Direct)_linear_positive'] = cumulative_op_positive[row['region']] / (months_in_period - building_time)

        # Distribute CPY (Direct)_linear_negative jobs (accumulate across periods)
        if row['region'] not in cumulative_cpy_negative:
            cumulative_cpy_negative[row['region']] = 0
        cumulative_cpy_negative[row['region']] += row['CPY (Direct)_linear_negative']
        if cumulative_cpy_negative[row['region']] < 0:
            temp_df['CPY (Direct)_linear_negative'] = cumulative_cpy_negative[row['region']] / months_in_period

        # Distribute Op FTE (Direct)_linear_negative jobs (accumulate across periods)
        if row['region'] not in cumulative_op_negative:
            cumulative_op_negative[row['region']] = 0
        cumulative_op_negative[row['region']] += row['Op FTE (Direct)_linear_negative']
        if cumulative_op_negative[row['region']] < 0:
            temp_df['Op FTE (Direct)_linear_negative'] = cumulative_op_negative[row['region']] / months_in_period

        # Append the temporary DataFrame to the accumulated_jobs DataFrame
        accumulated_jobs = pd.concat([accumulated_jobs, temp_df], ignore_index=True)

    # Create a national summary by grouping by the same attributes and summing the values
    national_jobs = accumulated_jobs.groupby(['variable', 'model', 'scenario', 'time']).sum().reset_index()
    national_jobs['region'] = 'National'

    # Ensure national summary has same columns as the original accumulated_jobs DataFrame
    national_jobs = national_jobs[accumulated_jobs.columns]

    # Append the national summary to the accumulated_jobs DataFrame
    accumulated_jobs = pd.concat([accumulated_jobs, national_jobs], ignore_index=True)

    # Reset the index
    accumulated_jobs.reset_index(drop=True, inplace=True)

    return accumulated_jobs


def plot_jobs_time_series(df, regions):
    """Plot the time series of jobs created and destroyed for a subset of regions."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    for region in regions:
        region_data = df[df['region'] == region]
        region_data_grouped = region_data.groupby('time').sum().reset_index()
        
        axs[0, 0].plot(region_data_grouped['time'], region_data_grouped['CPY (Direct)_linear_positive'], label=f'{region} CPY Created')
        axs[1, 0].plot(region_data_grouped['time'], region_data_grouped['CPY (Direct)_linear_negative'], label=f'{region} CPY Destroyed')
        
        axs[0, 1].plot(region_data_grouped['time'], region_data_grouped['Op FTE (Direct)_linear_positive'], label=f'{region} Op Created')
        axs[1, 1].plot(region_data_grouped['time'], region_data_grouped['Op FTE (Direct)_linear_negative'], label=f'{region} Op Destroyed')

    axs[0, 0].set_title('CPY Jobs Created Over Time')
    axs[0, 0].set_ylabel('Number of Jobs Created')
    axs[0, 0].legend()
    
    axs[1, 0].set_title('CPY Jobs Destroyed Over Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Number of Jobs Destroyed')
    axs[1, 0].legend()
    
    axs[0, 1].set_title('Op Jobs Created Over Time')
    axs[0, 1].set_ylabel('Number of Jobs Created')
    axs[0, 1].legend()
    
    axs[1, 1].set_title('Op Jobs Destroyed Over Time')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Number of Jobs Destroyed')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
    

accumulated_jobs = make_accumulate_jobs(df, building_time=12)
# Plot jobs time series for a subset of regions
plot_jobs_time_series(accumulated_jobs, ['British Columbia.a', 'National'])






plot_jobs_time_series(accumulated_jobs, ['British Columbia.a', 'Alberta'])
plot_jobs_time_series(accumulated_jobs, ['National'])


# Call the function to create the accumulated jobs DataFrame


accumulate_jobs.head()


# Display the resulting DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Accumulated Jobs Data", dataframe=accumulated_jobs)


# Call the function to create the accumulated jobs DataFrame
accumulated_jobs = accumulate_jobs(df, building_time=12)

accumulated_jobs[(accumulated_jobs["region"] == "National") & (accumulated_jobs["time"] == \
                                                       pd.Timestamp("2025-01-31"))]

accumulated_jobs["time"].unique()

# Display the resulting DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Accumulated Jobs Data", dataframe=accumulated_jobs)



# Create a DataFrame to hold the monthly jobs data
monthly_jobs = pd.DataFrame()

# Loop through each row in the original DataFrame
for index, row in df.iterrows():
    start_year = row['time']
    end_year = start_year + 4  # Each period is 5 years
    
    # Generate date range for the period
    date_range = pd.date_range(start=f'{start_year}-01', end=f'{end_year}-12', freq='M')
    
    # Calculate monthly jobs created and destroyed
    cpy_positive_monthly = row['CPY (Direct)_linear_positive'] / months_in_period
    cpy_negative_monthly = row['CPY (Direct)_linear_negative'] / months_in_period
    op_fte_positive_monthly = row['Op FTE (Direct)_linear_positive'] / months_in_period
    op_fte_negative_monthly = row['Op FTE (Direct)_linear_negative'] / months_in_period
    
    # Create a temporary DataFrame for the current row
    temp_df = pd.DataFrame({
        'region': row['region'],
        'variable': row['variable'],
        'model': row['model'],
        'scenario': row['scenario'],
        'time': date_range,
        'CPY (Direct)_linear_positive_monthly': cpy_positive_monthly,
        'CPY (Direct)_linear_negative_monthly': cpy_negative_monthly,
        'Op FTE (Direct)_linear_positive_monthly': op_fte_positive_monthly,
        'Op FTE (Direct)_linear_negative_monthly': op_fte_negative_monthly
    })
    
    # Append the temporary DataFrame to the monthly_jobs DataFrame
    monthly_jobs = pd.concat([monthly_jobs, temp_df], ignore_index=True)

# Create a national summary by grouping by the same attributes and summing the values
national_jobs = monthly_jobs.groupby(['variable', 'model', 'scenario', 'time']).sum().reset_index()
national_jobs['region'] = 'National'

# Ensure national summary has same columns as the original monthly_jobs DataFrame
national_jobs = national_jobs[monthly_jobs.columns]

# Append the national summary to the monthly_jobs DataFrame
monthly_jobs = pd.concat([monthly_jobs, national_jobs], ignore_index=True)

# Reset the index
monthly_jobs.reset_index(drop=True, inplace=True)

monthly_jobs[monthly_jobs["region"]  == "National"]
monthly_jobs[(monthly_jobs["region"] == "National") & (monthly_jobs["time"] == \
                                                       pd.Timestamp("2025-01-31"))]

# Display the filtered row
print("Unique regions:", unique_regions)
print("Unique scenarios:", unique_scenarios)


# Aggregating across regions and time
aggregated_df = df.groupby(['region', 'time']).agg({
    'CPY (Direct)_linear': 'sum',
    'Op FTE (Direct)_linear': 'sum'
}).reset_index()

# Aggregating across all regions and time
total_aggregation = aggregated_df.agg({
    'CPY (Direct)_linear': 'sum',
    'Op FTE (Direct)_linear': 'sum'
}).reset_index()

print("Total aggregation across all regions and time:")
print(total_aggregation)

population = 5e6 (population of british colombia)

df["CPY (Direct)_linear"]

df["Op FTE (Direct)_linear"]


print(df[['CPY (Direct)_linear', 'CPY (Direct)_linear_positive', 'CPY (Direct)_linear_negative', 
          'Op FTE (Direct)_linear', 'Op FTE (Direct)_linear_positive', 'Op FTE (Direct)_linear_negative']])



# Filter the dataframe for "British Columbia.a"
bc_df = df[df['region'] == "British Columbia.a"]

# Aggregate by year for British Columbia for the new columns
bc_aggregate_by_year = bc_df.groupby('time').agg({
    'CPY (Direct)_linear_positive': 'sum',
    'CPY (Direct)_linear_negative': 'sum',
    'Op FTE (Direct)_linear_positive': 'sum',
    'Op FTE (Direct)_linear_negative': 'sum'
}).reset_index()

print("Aggregate by year for British Columbia:")
print(bc_aggregate_by_year)

# Aggregate by year for each region
region_aggregates = {}
for region in unique_regions:
    region_df = df[df['region'] == region]
    region_aggregate_by_year = region_df.groupby('time').agg({
        'CPY (Direct)_linear_positive': 'sum',
        'CPY (Direct)_linear_negative': 'sum',
        'Op FTE (Direct)_linear_positive': 'sum',
        'Op FTE (Direct)_linear_negative': 'sum'
    }).reset_index()
    region_aggregates[region] = region_aggregate_by_year


region_aggregates[unique_regions[0]]

# Print aggregates for each region
for region, aggregate in region_aggregates.items():
    print(f"Aggregate by year for {region}:")
    print(aggregate)



# Filter the dataframe for "British Columbia.a"
bc_df = df[df['region'] == "British Columbia.a"]

# Aggregate across time for British Columbia
bc_aggregate = bc_df.groupby('time').agg({
    'CPY (Direct)_linear': 'sum',
    'Op FTE (Direct)_linear': 'sum'
}).reset_index()

# Aggregate across all regions and time
total_aggregate = df.groupby('region').agg({
    'CPY (Direct)_linear': 'sum',
    'Op FTE (Direct)_linear': 'sum'
}).reset_index()

# Find the region with the biggest aggregate of each type
max_cpy_direct_linear_region = total_aggregate.loc[total_aggregate['CPY (Direct)_linear'].idxmax()]['region']
max_op_fte_direct_linear_region = total_aggregate.loc[total_aggregate['Op FTE (Direct)_linear'].idxmax()]['region']

print("Aggregate for British Columbia:")
print(bc_aggregate)

print("\nRegion with the biggest aggregate CPY (Direct)_linear:")
print(max_cpy_direct_linear_region)

print("\nRegion with the biggest aggregate Op FTE (Direct)_linear:")
print(max_op_fte_direct_linear_region)