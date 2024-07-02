Files to prepare Canada data
make_Canada_timeseries_2015-2023.py will output u_v_gdp_seasonal_barometer.csv, this file will be used for the target demand as well as to compare results. It has unemployment, vacancies seasonally adjusted. It also has the gdp and business barometer filters

make_shocks.py will output shock_timeseries_region_tech.csv which is the shock time series at the technology level

map_tech_to_occ.py will output technologies_occ.csv which maps technologies to occupations. This will be used to translate the shock time series to occupations

map_shocks_to_occ.pu maps the shocks time series from technologies to occupation it outputs both the occ_mobility_fromusa.csv and scenario_{region}, so it gives a scenario for each region of Canada

in utils.py the function network_and_scenario will take files from above as input and make target demand for scenarios
