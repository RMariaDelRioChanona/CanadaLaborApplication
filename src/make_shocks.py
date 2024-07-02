import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
from matplotlib.ticker import MaxNLocator

def distribute_jobs_monthly(df):
    # Define the function to generate monthly date ranges
    def generate_monthly_range(start_year, end_year):
        return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')
    
    # Define important years to manage period boundaries correctly
    important_years = [2021, 2025, 2030, 2035, 2040, 2045, 2050]

    # List to store the new entries
    data = []

    # Job columns and their descriptive names
    job_columns = [
        ('CPY (Direct)_linear_positive', 'CPY_created_jobs'),
        ('CPY (Direct)_linear_negative', 'CPY_destroyed_jobs'),
        ('Op FTE (Direct)_linear_positive', 'Op_created_jobs'),
        ('Op FTE (Direct)_linear_negative', 'Op_destroyed_jobs')
    ]

    # Iterate over the DataFrame
    for _, row in df.iterrows():
        start_year = int(row['time'])
        if start_year == 2021:
            end_year = 2025
        else:
            start_year += 1  # Adjust start year for next period
            end_year = start_year + 4

        # Generate the monthly range for each period
        monthly_range = generate_monthly_range(start_year, end_year)

        # Distribute jobs monthly for each type
        for job_column, job_type in job_columns:
            job_value = row[job_column]
            job_per_month_acc = job_value / len(monthly_range)  # Distribute evenly across all months

            for date in monthly_range:
                data.append({
                    'region': row['region'],
                    'variable': row['variable'],
                    'model': row['model'],
                    'scenario': row['scenario'],
                    'time': date.strftime('%Y-%m'),  # Format as year-month
                    'unit': row['unit'],
                    'jobs': job_per_month_acc,
                    'job_type': job_type
                })



    monthly_jobs_df = pd.DataFrame(data)
    # Group by the relevant columns and sum the jobs for all regions
    national_jobs = monthly_jobs_df.groupby(['variable', 'model', 'scenario', 'time', 'job_type', 'unit'], as_index=False).agg({'jobs': 'sum'})

    # Add a 'region' column filled with 'National'
    national_jobs['region'] = 'National'

    # Concatenate the national_jobs DataFrame back to the original monthly_jobs_df
    updated_monthly_jobs_df = pd.concat([monthly_jobs_df, national_jobs], ignore_index=True)

    # Optionally, you might want to sort the DataFrame by time or other columns to maintain order
    updated_monthly_jobs_df = updated_monthly_jobs_df.sort_values(by=['region', 'variable', 'model', 'scenario', 'job_type', 'time'])

    # Convert the list to DataFrame
    return updated_monthly_jobs_df

# Prepare your initial DataFrame (this setup part is done once outside the function)
path_data = "../data/copper/"
file_zach = "net_new_cap_w_jobs.csv"
df = pd.read_csv(path_data + file_zach)
df = df[['region', 'variable', 'model', 'scenario', 'time', 'unit', 'value',
         'CPY (Direct)_linear', 'Op FTE (Direct)_linear']]
df['CPY (Direct)_linear_positive'] = df['CPY (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['CPY (Direct)_linear_negative'] = df['CPY (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)
df['Op FTE (Direct)_linear_positive'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x >= 0 else 0)
df['Op FTE (Direct)_linear_negative'] = df['Op FTE (Direct)_linear'].apply(lambda x: x if x <= 0 else 0)

df = df.groupby(['region', 'variable', 'model', 'scenario', 'time', 'unit'], as_index=False).sum()


df['CPY (Direct)_linear_positive'].sum()/1e6
df['Op FTE (Direct)_linear_positive'].sum()/1e6

# Call the function
monthly_jobs_df = distribute_jobs_monthly(df)

# ['CPY_created_jobs', 'CPY_destroyed_jobs', 'Op_created_jobs', 'Op_destroyed_jobs']

monthly_jobs_df['jobs'][(monthly_jobs_df['region'] == 'National') & \
                        (monthly_jobs_df['job_type']=='CPY_created_jobs')].sum()/1e6
monthly_jobs_df['jobs'][(monthly_jobs_df['region'] == 'National') \
                        & (monthly_jobs_df['job_type']=='Op_created_jobs')].sum()/1e6



monthly_jobs_df['variable'].unique()
monthly_jobs_df['region'].unique()


### Now make Operation jobs cumulative
# Step 1: Filter out the 'Op' job types
op_jobs_df = monthly_jobs_df[monthly_jobs_df['job_type'].isin(['Op_created_jobs', 'Op_destroyed_jobs'])]

# Step 2: Sort by 'time' to ensure correct order for cumulative sum
op_jobs_df = op_jobs_df.sort_values(by=['region', 'variable', 'model', 'scenario', 'job_type', 'time'])

# Step 3: Apply cumulative sum on the 'jobs' column within each group
op_jobs_df['jobs'] = op_jobs_df.groupby(['region', 'variable', 'model', 'scenario', 'job_type'])['jobs'].cumsum()

# Step 4: Extract the non-Op job types to concatenate them back together
non_op_jobs_df = monthly_jobs_df[~monthly_jobs_df['job_type'].isin(['Op_created_jobs', 'Op_destroyed_jobs'])]

# Step 5: Concatenate back the modified Op jobs data with the non-Op jobs data
updated_monthly_jobs_df = pd.concat([non_op_jobs_df, op_jobs_df], ignore_index=True)

# Sorting to restore any order if necessary
updated_monthly_jobs_df = updated_monthly_jobs_df.sort_values(by=['region', 'variable', 'model', 'scenario', 'job_type', 'time'])


updated_monthly_jobs_df.to_csv(path_data + "shock_timeseries_region_tech_cap_op.csv", index=False)

# Optional: Check the output
print(updated_monthly_jobs_df.head())

def plot_job_data(region, variable):
    # Filter the data for the specified region and variable
    filtered_data = updated_monthly_jobs_df[(updated_monthly_jobs_df['region'] == region) &
                                            (updated_monthly_jobs_df['variable'] == variable)]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=120)  # Adjust size and resolution as needed
    fig.suptitle(f"Job Creation and Destruction Over Time in {region} for {variable}", fontsize=16)

    # Define a helper to plot on a given axis
    def plot_subplot(ax, job_type, title, color):
        sns.lineplot(x='time', y='jobs', data=filtered_data[filtered_data['job_type'] == job_type], ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Jobs')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure that x-axis labels are integer (yearly)
        plt.xticks(rotation=45)  # Rotate x-ticks for better readability

    # Plotting each job type
    plot_subplot(axs[0, 0], 'CPY_created_jobs', 'CPY Created Jobs', 'blue')
    plot_subplot(axs[0, 1], 'CPY_destroyed_jobs', 'CPY Destroyed Jobs', 'red')
    plot_subplot(axs[1, 0], 'Op_created_jobs', 'Op Created Jobs', 'green')
    plot_subplot(axs[1, 1], 'Op_destroyed_jobs', 'Op Destroyed Jobs', 'orange')

    # Adjust layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which to fit subplots

    # Show the plot
    plt.show()

# Example usage of the function
plot_job_data('British Columbia.a', 'Electricity|Wind|Offshore')
plot_job_data('British Columbia.a', 'Electricity|Solar|PV|Open Land')


plot_job_data('National', 'Electricity|Wind|Offshore')
plot_job_data('National', 'Transmission')

plot_job_data('National', 'Electricity|Coal|w/o CCS')

plot_job_data('National', 'Electricity|Solar|PV|Open Land')


###### Now just make created jobs and destroyed jobs

updated_monthly_jobs_df['job_type'] = updated_monthly_jobs_df['job_type'].replace({
    'CPY_created_jobs': 'jobs',
    'Op_created_jobs': 'jobs',
    'CPY_destroyed_jobs': 'jobs',
    'Op_destroyed_jobs': 'jobs'
})

# Group by all columns except 'jobs' and sum the 'jobs' values
df_grouped = updated_monthly_jobs_df.groupby(['region', 'variable', 'model', 'scenario', 'time', 'unit', 'job_type']).sum().reset_index()

# df_grouped['jobs'][(df_grouped['region'] == "National")].sum()/1e6

df_grouped['jobs'][(df_grouped['region'] == "National") & (df_grouped['time'] == "2029-12")].sum()
df_grouped['jobs'][(df_grouped['region'] == "National") & (df_grouped['time'] == "2050-12")].sum()


plt.plot(df_grouped[(df_grouped['region'] == "National")]['time'], \
         df_grouped[(df_grouped['region'] == "National")]['jobs'])
plt.show()

df_grouped[(df_grouped['region'] == "National")]


# df_grouped[(df_grouped['region'] == "National") & df_grouped[]]


# NOTE the one below aggregates and is the final one that matters
df_grouped.to_csv(path_data + "shock_timeseries_region_tech.csv", index=False)



def plot_jobs_by_region(region, df_grouped):
    # Filter the DataFrame for the given region
    df_region = df_grouped[df_grouped['region'] == region]

    # Group by time and sum up the jobs
    time_series = df_region.groupby('time')['jobs'].sum()

    # Plotting the time series data
    plt.figure(figsize=(10, 6))
    plt.plot(time_series.index, time_series.values, marker='o', linestyle='-')
    plt.title(f'Total Jobs Over Time in {region}')
    plt.xlabel('Time')
    plt.ylabel('Total Jobs')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate dates for better visibility
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()

# Example usage:
plot_jobs_by_region('Alberta.a', df_grouped)


plot_jobs_by_region('National', df_grouped)

# # Example: Filter data for 'Alberta.a' and 'Electricity|Biomass|w/o CCS'
# filtered_data = monthly_jobs_df[(monthly_jobs_df['region'] == 'British Columbia.a') &
#                                 (monthly_jobs_df['variable'] == 'Electricity|Wind|Offshore')]


# filtered_data = updated_monthly_jobs_df[(updated_monthly_jobs_df['region'] == 'British Columbia.a') &
#                                 (updated_monthly_jobs_df['variable'] == 'Electricity|Wind|Offshore')]

# # Create a figure with subplots
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid, adjust the size as needed
# fig.suptitle("Job Creation and Destruction Over Time", fontsize=16)

# # Plot each job category in a subplot
# sns.lineplot(x='time', y='jobs', data=filtered_data[filtered_data['job_type'] == 'CPY_created_jobs'], ax=axs[0, 0], color='blue')
# axs[0, 0].set_title('CPY Created Jobs')
# axs[0, 0].set_ylabel('Number of Jobs')
# axs[0, 0].set_xlabel('Time')

# sns.lineplot(x='time', y='jobs', data=filtered_data[filtered_data['job_type'] == 'CPY_destroyed_jobs'], ax=axs[0, 1], color='red')
# axs[0, 1].set_title('CPY Destroyed Jobs')
# axs[0, 1].set_ylabel('Number of Jobs')
# axs[0, 1].set_xlabel('Time')

# sns.lineplot(x='time', y='jobs', data=filtered_data[filtered_data['job_type'] == 'Op_created_jobs'], ax=axs[1, 0], color='green')
# axs[1, 0].set_title('Op Created Jobs')
# axs[1, 0].set_ylabel('Number of Jobs')
# axs[1, 0].set_xlabel('Time')

# sns.lineplot(x='time', y='jobs', data=filtered_data[filtered_data['job_type'] == 'Op_destroyed_jobs'], ax=axs[1, 1], color='orange')
# axs[1, 1].set_title('Op Destroyed Jobs')
# axs[1, 1].set_ylabel('Number of Jobs')
# axs[1, 1].set_xlabel('Time')

# # Automatically adjust subplot params so that the subplot(s) fits in to the figure area
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # Show the plot
# plt.show()


# # Create a figure with subplots
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid
# fig.suptitle("Job Creation and Destruction Over Time", fontsize=16)

# # Define a helper function to plot data
# def plot_job_data(ax, job_type, title, color):
#     subset = filtered_data[filtered_data['job_type'] == job_type]
#     ax.plot(subset['time'], subset['jobs'], color=color, marker='o', linestyle='-')
#     ax.set_title(title)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Number of Jobs')
#     ax.tick_params(axis='x', rotation=45)  # Rotate date labels for better visibility

# # Plotting each type of job data
# plot_job_data(axs[0, 0], 'CPY_created_jobs', 'CPY Created Jobs', 'blue')
# plot_job_data(axs[0, 1], 'CPY_destroyed_jobs', 'CPY Destroyed Jobs', 'red')
# plot_job_data(axs[1, 0], 'Op_created_jobs', 'Op Created Jobs', 'green')
# plot_job_data(axs[1, 1], 'Op_destroyed_jobs', 'Op Destroyed Jobs', 'orange')

# # Adjust layout to not overlap and make space for title
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Show the plot
# plt.show()


# # # Set the style of seaborn
# # sns.set(style="whitegrid")





# monthly_jobs_df[monthly_jobs_df['job_type'] == 'Op_created_jobs'][monthly_jobs_df['time'] == '2025-01'].head(20)


# monthly_jobs_df[monthly_jobs_df['time'] == '2025-01'].iloc[400:420]

















# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# years= [2021, 2025, 2030, 2035, 2040, 2045, 2050]

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


# # NOTE this aggregation is important, since the original file has repeated entries. Discuss with Zach
# # Aggregate rows with the same region, variable, and time
# df = df.groupby(['region', 'variable', 'model', 'scenario', 'time', 'unit'], as_index=False).sum()

# df.columns


# cumulative_jobs = {
#         'CPY_negative': 0,
#         'Op_positive': 0,
#         'Op_negative': 0
#     }


# df['variable'].unique()
# df['region'].unique()

# ### 
# df = df[(df['region'].isin(['Alberta.a', 'British Columbia.a'])) & \
#         df['variable'].isin(['Electricity|Biomass|w/o CCS', 'Electricity|Gas|CC|w/ CCS']) ]


# # Function to generate monthly date ranges
# def generate_monthly_range(start_year, end_year):
#     return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

# important_years = [2021, 2025, 2030, 2035, 2040, 2045, 2050]

# # List to store data
# data = []

# # Job columns and their descriptive names
# job_columns = [
#     ('CPY (Direct)_linear_positive', 'CPY_created_jobs'),
#     ('CPY (Direct)_linear_negative', 'CPY_destroyed_jobs'),
#     ('Op FTE (Direct)_linear_positive', 'Op_created_jobs'),
#     ('Op FTE (Direct)_linear_negative', 'Op_destroyed_jobs')
# ]

# # Iterate over the dataframe
# for _, row in df.iterrows():
#     start_year = int(row['time'])
#     if start_year == 2021:
#         end_year = 2025  # The period starts in 2021 and ends in December 2025
#     else:
#         start_year += 1  # Start from the next year after the end of the previous period
#         end_year = start_year + 4  # Subsequent periods are 5 years, starting from January after last period ended


#     # Generate monthly range for each period
#     monthly_range = generate_monthly_range(start_year, end_year)
    
#     if start_year in important_years or end_year in important_years:
#         print(f"Start of period {start_year}, End of period {end_year}, Processing row {index}")


#     # Distribute jobs monthly for each type
#     for job_column, job_type in job_columns:
#         job_value = row[job_column]
#         job_per_month_acc = job_value / len(monthly_range)  # Distribute evenly across all months

#         for date in monthly_range:
#             data.append({
#                 'region': row['region'],
#                 'variable': row['variable'],
#                 'model': row['model'],
#                 'scenario': row['scenario'],
#                 'time': date.strftime('%Y-%m'),  # Format as year-month
#                 'unit': row['unit'],
#                 'jobs': job_per_month_acc,
#                 'job_type': job_type
#             })
#             if job_type == 'CPY_positive_createdjobs':
#                 if date.strftime('%Y-%m').split('-')[0] in ['2021', '2025', '2030', '2035', '2040', '2045', '2050']:
#                     print(f"Adding entry for {date.strftime('%Y-%m')} with jobs {job_per_month_acc}, job type {job_type}")


# # Convert list to DataFrame
# monthly_jobs_df = pd.DataFrame(data)

# monthly_jobs_df[monthly_jobs_df['time'] == '2034-01']


# region_tech_data = monthly_jobs_df[(monthly_jobs_df['region'] == 'Alberta.a') &\
#                                     (monthly_jobs_df['variable'] == 'Electricity|Biomass|w/o CCS')]


# job_data = region_tech_data[region_tech_data['job_type'] == 'CPY_positive_createdjobs']

# job_data[job_data['time'] == '2025-01']
# job_data[job_data['time'] == '2025-02']
# job_data[job_data['time'] == '2030-01']


# job_data[job_data['time'] == '2026-01']

# job_data[job_data['time'] == '2036-01']

# df_test = monthly_jobs_df[monthly_jobs_df["job_type"] == "CPY_positive_createdjobs" ]


# data = []
# for _, row in df.iterrows():
#     for period_start in years:
#         if row['region'] == 'Alberta.a':
#             print(row['variable'],period_start)
#         if period_start == 2021:
#             monthly_range = generate_monthly_range(period_start, period_start + 4)
#         else:
#             monthly_range = generate_monthly_range(period_start, period_start + 5)
#         months_in_period = len(monthly_range)
#         bulk_months = bulk_years * 12
#         job_columns = [
#             ('CPY (Direct)_linear_positive', 'CPY_positive_bulk'),
#             ('CPY (Direct)_linear_negative', 'CPY_negative_accumulated'),
#             ('Op FTE (Direct)_linear_positive', 'Op_positive_accumulated'),
#             ('Op FTE (Direct)_linear_negative', 'Op_negative_accumulated')
#         ]

#         for job_column, job_type in job_columns:
#             job_value = row[job_column]
            
#             if job_type == 'CPY_positive_bulk':
#                 # Bulk jobs
#                 jobs_per_month_bulk = job_value / bulk_months
#                 for i, date in enumerate(monthly_range):
#                     jobs_bulk = jobs_per_month_bulk if (date.year - period_start) < bulk_years else 0
#                     bulk_entry = {
#                         'region': row['region'],
#                         'variable': row['variable'],
#                         'model': row['model'],
#                         'scenario': row['scenario'],
#                         'time': date,
#                         'unit': row['unit'],
#                         'jobs': jobs_bulk,
#                         'job_type': job_type
#                     }
#                     data.append(bulk_entry)



# columns = ['time', 'CPY (Direct)_linear_negative', 'Op FTE (Direct)_linear_positive', 'Op FTE (Direct)_linear_negative']
# monthly_data = pd.DataFrame(columns=columns)

# # Function to calculate the number of months between years
# def months_between(start_year, end_year):
#     return (end_year - start_year) * 12


# # List to hold the data
# data_list = []

# # Populate the list with monthly data
# for index, row in df.iterrows():
#     start_year = int(row['time'])
#     if start_year == 2050:
#         continue  # Adjust as needed if there's no period after 2050 in your dataset
#     end_year = start_year + (5 if start_year != 2021 else 4)  # 4 years for the first period, 5 for the others
#     months = months_between(start_year, end_year)
#     for month in range(months):
#         data_list.append({
#             'time': f"{start_year + month // 12}-{(month % 12) + 1}",
#             'CPY (Direct)_linear_negative': row['CPY (Direct)_linear_negative'] / months,
#             'Op FTE (Direct)_linear_positive': row['Op FTE (Direct)_linear_positive'] / months,
#             'Op FTE (Direct)_linear_negative': row['Op FTE (Direct)_linear_negative'] / months
#         })

# # Convert list to DataFrame
# monthly_data = pd.DataFrame(data_list)