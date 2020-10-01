import numpy as np
import pandas as pd
from datetime import datetime

## This works
# country = 'Austria'
country = 'France'
# country = 'Germany'
# country = 'Spain'
# country = 'England'
# country = 'Scotland'
# country = 'Northern Ireland'

## Following still not considered
# country = 'Scotland'
# country = 'Belgium'
# country = 'Denmark'
# country = 'Netherlands'
# country = 'Norway'
# country = 'Portugal'
# country = 'South Korea'
# country = 'Sweden'
# country = 'Ukraine'
# country = 'United States'

start_date = "1Mar"
end_date = "31Aug"

subfolder = f"{country.lower()}_inference_data_{start_date}_to_{end_date}"
# ############################ Reading contact matrix data #######################
# These are contact matrices for 5 years age groups, up to the age of 75, and a single age group from 75+. By comparing
# with the plots in the original Prem's work, I think each column represents the individual, and each row the number of
# contacts that the individual has with each age group.
# If you convert the dataframe to numpy array, the first index refers to the individual, and the second to the contacts
# with other age groups. This is what we want in our formulation.
# Note: entries are age-stratified expected number of contacts per day.
if country in ['Austria', 'France', 'Germany', 'Italy']:
    fileno, headertype = '1', 0
else:
    fileno, headertype = '2', None

if country in ['England', 'Scotland', 'Northern Ireland']:
    country_contact = 'United Kingdom of Great Britain'
else:
    country_contact = country
contact_matrix_all_locations = pd.read_excel(
    "data/contact_matrices_152_countries/MUestimates_all_locations_" + fileno + ".xlsx"
    , sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_home = pd.read_excel("data/contact_matrices_152_countries/MUestimates_home_" + fileno + ".xlsx",
                                    sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_school = pd.read_excel("data/contact_matrices_152_countries/MUestimates_school_" + fileno + ".xlsx",
                                      sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_work = pd.read_excel("data/contact_matrices_152_countries/MUestimates_work_" + fileno + ".xlsx",
                                    sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_other_locations = pd.read_excel(
    "data/contact_matrices_152_countries/MUestimates_other_locations_" + fileno + ".xlsx",
    sheet_name=country_contact, header=headertype).to_numpy()
########################################################################################################################
################################ United Nation World Population Data by age #############################################
if country in ['England', 'Scotland', 'Northern Ireland']:
    country_UN_data = 'United Kingdom of Great Britain and Northern Ireland'
else:
    country_UN_data = country

UN_Pop_Data = pd.read_csv("data/UNdata_Export_20200820_181107223.csv")
## Choose both the sexes
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Sex"] == 'Both Sexes']
## Choose the total area
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Area"] == 'Total']
# Choose the country
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Country or Area"] == country_UN_data]
# Choose the most recent year
if UN_Pop_Data[UN_Pop_Data["Year"] == '2018'].empty:
    UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Year"] == 2018]
else:
    UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Year"] == '2018']
Country_age_data_list = []
age_list = ['0 - 4', '5 - 9', '10 - 14', '15 - 19', '20 - 24', '25 - 29', '30 - 34', '35 - 39', '40 - 44', '45 - 49',
            '50 - 54', '55 - 59', '60 - 64', '65 - 69', '70 - 74', 'Total']
age_list_lb = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
for x in age_list:
    if (UN_Pop_Data['Age'] == x).any():
        Country_age_data_list.append(UN_Pop_Data[UN_Pop_Data['Age'] == x]['Value'].iloc[0])
Country_age_data_list[-1] = Country_age_data_list[-1] - sum(Country_age_data_list[:15])
Country_age_weight = Country_age_data_list / sum(Country_age_data_list)

########################################################################################################################
################################ Countrywise Data for age-structured death #############################################
if country == 'France':
    sheetname, rows = 'SpF_by age and sex_HospitalData', 12
elif country == 'England & Wales':
    ## Weekly death data which accounts for deaths in hospital, hospice and other places
    sheetname, rows = 'ONS_WeeklyOccurrenceDeaths', 19
elif country == 'England':
    sheetname, rows = 'NHS_Daily_Data', 7
elif country == 'Scotland':
    sheetname, rows = 'NRS_Age_&_Sex', 7
elif country == 'Spain':
    sheetname, rows = 'MSCBS_Data', 9
elif country == 'Germany':
    sheetname, rows = 'Daily Report RKI_Data', 10
elif country == 'Austria':
    sheetname, rows = 'EMS_Data', 10
elif country == 'Belgium':
    sheetname, rows = 'Deaths_by_occurrence', '6'

if country == 'France':
    file = "data/Deaths-Age-Sex_Covid-19_France_01-09.xlsx"
    data = pd.read_excel(file, sheet_name=sheetname, skiprows=[0, 1, 2, 3, 4], nrows=rows, header=None)
    days = data.iloc[0, [i * 7 for i in range(1, int(len(data.columns) / 7))]]
    dates_yday = [(datetime.strptime(str(x)[:10].replace('.', '-'), '%Y-%m-%d').timetuple().tm_yday) for x in days]
    death_data = data.iloc[2:, [i * 7 + 5 for i in range(1, int(len(data.columns) / 7))]].to_numpy()
    age_groups = data.iloc[2:, 0].to_numpy()
    age_groups_lb = [int(x.replace('+', '-').split("-")[0]) for x in age_groups]
    total_population_age_group = data.iloc[2:, 1].to_numpy()

## We choose 5 age groups (same as England data) to match our model structure
age_groups = ['0-19', '20-39', '40-59', '60-79', '80+']
age_groups_lb = [0, 20, 40, 60, 80]
death_data_5 = np.zeros(shape=(5, death_data.shape[1]))
# age-group: '0-19'
death_data_5[0, :] = death_data[0, :]
# age-group: '20-39'
death_data_5[1, :] = np.sum(death_data[1:3, :], axis=0)
# age-group: '40-59'
death_data_5[2, :] = np.sum(death_data[3:5, :], axis=0)
# age-group: '60-79'
death_data_5[3, :] = np.sum(death_data[5:7, :], axis=0)
# age-group: '80+'
death_data_5[4, :] = np.sum(death_data[7:9, :], axis=0)

# Consider modified data as death data
death_data = death_data_5


# print(age_groups, age_groups_lb, death_data.shape)

## ## Modify contact matrix using the age_groups_lb ###
def transform_contact_matrix(contact_matrix, Country_age_data_list):
    # Remember that first index represents individual, second age of contact.

    # define region populations
    region_pop_0_19 = np.sum(Country_age_data_list[0:4])  # pop_df[pop_df["Name"] == region]["0-19"].to_numpy()
    region_pop_20_39 = np.sum(Country_age_data_list[4:8])  # pop_df[pop_df["Name"] == region]["20-39"].to_numpy()
    region_pop_40_59 = np.sum(Country_age_data_list[8:12])  # pop_df[pop_df["Name"] == region]["40-59"].to_numpy()
    region_pop_60_79 = np.sum(Country_age_data_list[12:16])  # pop_df[pop_df["Name"] == region]["60-79"].to_numpy()
    region_pop_70_79 = np.sum(Country_age_data_list[15:17])  # pop_df[pop_df["Name"] == region]["70-79"].to_numpy()
    region_pop_75_79 = Country_age_data_list[-2]  # pop_df[pop_df["Name"] == region]["75-79"].to_numpy()
    region_pop_80 = Country_age_data_list[-1]  # pop_df[pop_df["Name"] == region]["80+"].to_numpy()

    # We first need to sum the age of contacts in blocks of 4; however, note that the last group in the contact
    # matrices is 70+, instead of 80+ as in the other work. We need therefore first to split the number of contacts
    # there in the ones for 70+ and the ones for 80+; we assume they are equally spread.

    contact_matrix_1 = np.zeros((16, 17))
    # contact_matrix_1 = np.zeros((15, 16))
    contact_matrix_1[:, 0:-1] = contact_matrix

    contact_matrix_1[:, -1] = contact_matrix_1[:, -2] * region_pop_80 / (region_pop_80 + region_pop_75_79)
    contact_matrix_1[:, -2] = contact_matrix_1[:, -2] * region_pop_75_79 / (region_pop_80 + region_pop_75_79)

    # Now we need to sum the number of contacts in groups of 4:
    contact_matrix_2 = np.zeros((16, 5))
    # contact_matrix_2 = np.zeros((15, 5))  # row 14 now represents 70-79.

    for i in range(4):
        contact_matrix_2[:, i] = np.sum(contact_matrix_1[:, i * 4:(i + 1) * 4], axis=1)
    contact_matrix_2[:, -1] = contact_matrix_1[:, -1]  # 80+ group

    # We now need to average over the age of individual, taking into account the number of people in each of the 5 years groups.
    contact_matrix_3 = np.zeros((5, 5))

    for i in range(4):
        tot_population_20y_age_group = np.sum(Country_age_data_list[i * 4:(i + 1) * 4])
        for j in range(4):
            tot_population_5y_age_group = Country_age_data_list[i * 4 + j]
            # for k in range(5):
            #     tot_population_5y_age_group += pop_df[pop_df["Name"] == region][20 * i + 5 * j + k].to_numpy()
            #     tot_population_20y_age_group += pop_df[pop_df["Name"] == region][20 * i + 5 * j + k].to_numpy()

            if i == 3 and j == 3:  # ad-hoc fix for the case of contact matrix with 70+ column
                contact_matrix_3[i, :] += tot_population_5y_age_group * contact_matrix_2[i * 4 + j - 1, :]
            else:
                contact_matrix_3[i, :] += tot_population_5y_age_group * contact_matrix_2[i * 4 + j, :]

        contact_matrix_3[i, :] /= tot_population_20y_age_group

    contact_matrix_3[-1, :] = contact_matrix_2[-1,
                              :]  # for individuals in the 80+ age group, we assume the contacts are the same as in the 75+ age group

    return contact_matrix_3


contact_matrix_home = transform_contact_matrix(contact_matrix_home, Country_age_data_list)
contact_matrix_work = transform_contact_matrix(contact_matrix_work, Country_age_data_list)
contact_matrix_school = transform_contact_matrix(contact_matrix_school, Country_age_data_list)
contact_matrix_other_locations = transform_contact_matrix(contact_matrix_other_locations, Country_age_data_list)
contact_matrix_all_locations = contact_matrix_home + contact_matrix_work + contact_matrix_school + contact_matrix_other_locations

np.save('data/' + subfolder + '/contact_matrix_home.npy', contact_matrix_home)
np.save('data/' + subfolder + '/contact_matrix_work.npy', contact_matrix_work)
np.save('data/' + subfolder + '/contact_matrix_school.npy', contact_matrix_school)
np.save('data/' + subfolder + '/contact_matrix_other.npy', contact_matrix_other_locations)

total_population_age_group = [np.sum(Country_age_data_list[0:4]), np.sum(Country_age_data_list[4:8]),
                              np.sum(Country_age_data_list[8:12]), np.sum(Country_age_data_list[12:16]),
                              Country_age_data_list[-1]]
np.save('data/' + subfolder + f'/{country.lower()}_pop.npy', total_population_age_group)
####################### Final Output : Death_Data ###########################################
##### de-cumsum of death data ####
death_data = np.nan_to_num(np.fliplr(death_data))
dates_yday = np.flipud(dates_yday)
death_data_decum = np.zeros(shape=death_data.shape)
death_data_decum[:, 0] = death_data[:, 0]
for ind in range(1, death_data.shape[1]):
    death_data_decum[:, ind] = death_data[:, ind] - death_data[:, ind - 1]
## Death data is in cumulative format
Death_Data = np.concatenate((dates_yday[1:].reshape(1, -1), death_data_decum[:, 1:]))

hospitalized_data = pd.read_csv("data/hospital-data-covid.csv", header=0)
date = hospitalized_data['jour'].unique()
dates_yday = np.array(
    [int(datetime.strptime(str(x)[:10].replace('/', '-'), '%Y-%m-%d').timetuple().tm_yday) for x in date])
hospitalized = np.array([hospitalized_data.loc[hospitalized_data['jour'] == x, 'hosp'].sum() for x in date])
Other_Data = np.vstack((dates_yday, hospitalized))


##################################### Final Output: Other_data #########################################################

########################################################################################################################
################################## Merge Death and Confirmed/Admitted data ################################################################
def merging_function(death, confirmed):
    ## Keeping data only between 1st March and 31st August
    start_day, end_day = min(np.min(death[0, :]), np.min(confirmed[0, :]), 61), min(
        max(np.max(death[0, :]), np.max(confirmed[0, :])), 244)
    Data = [[ind] for ind in range(int(start_day), int(end_day) + 1)]
    for ind in range(len(Data)):
        if Data[ind][0] in death[0, :]:
            Data[ind] = Data[ind] + death[1:, np.where(death[0, :] == Data[ind][0])[0]].reshape(-1, ).tolist()
        else:
            for i in range(death.shape[0] - 1):
                Data[ind].append(None)
        if Data[ind][0] in confirmed[0, :]:
            Data[ind] = Data[ind] + confirmed[1:, np.where(confirmed[0, :] == Data[ind][0])[0]].reshape(-1, ).tolist()
        else:
            for i in range(confirmed.shape[0] - 1):
                Data[ind].append(None)

    return np.transpose(Data)


Observed_Data = merging_function(Death_Data, Other_Data).transpose()

# ## Keep Observed Data only from 1st March
# dynamics_start_day = int(datetime.strptime(str('2020-03-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
# index_dynamics_start_day = Observed_Data[0,:].tolist().index(dynamics_start_day)
# Observed_Data = np.delete(Observed_Data, np.arange(0, index_dynamics_start_day), axis=1)
#
# ## Keep Observed Data until 31st August
# dynamics_end_day = int(datetime.strptime(str('2020-08-31')[:10], '%Y-%m-%d').timetuple().tm_yday)
# index_dynamics_end_day = Observed_Data[0,:].tolist().index(dynamics_end_day) + 1
# last_day = Observed_Data[0,:].tolist().index(Observed_Data[0,-1]) + 1
# Observed_Data = np.delete(Observed_Data, np.arange(index_dynamics_end_day, last_day), axis=1).transpose()[:, 1:]
#
np.save('data/' + subfolder + '/observed_data.npy', Observed_Data[:, 1:])

###################################### Final Output: Observed_data ######################################################

########################################################################################################################
################################## Reading Google mobility data ################################################################
if country in ['England', 'Scotland', 'Northern Ireland', 'England & Wales']:
    country_mobility = 'United Kingdom'
    start_date_lockdown = 77  ## 17th March in yday format
else:
    country_mobility = country
    ## Load lockadown dates for each countroes
    lockdown_dates = pd.read_csv("data/Lockdown_Dates.csv")
    lockdown_dates = lockdown_dates[lockdown_dates["Level "] == 'National ']
    print(lockdown_dates[lockdown_dates["Countries and territories "] == ' ' + country + ' ']["Start date "])
    start_date_lockdown = int(datetime.strptime(
        lockdown_dates[lockdown_dates["Countries and territories "] == ' ' + country + ' ']["Start date "].iloc[0][:10],
        '%Y-%m-%d').timetuple().tm_yday)
    end_date_lockdown = int(datetime.strptime(
        lockdown_dates[lockdown_dates["Countries and territories "] == ' ' + country + ' ']["End date "].iloc[0][:10],
        '%Y-%m-%d').timetuple().tm_yday)

# url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
# url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=fbb340e43a0602e1"
# global_mobility = pd.read_csv(url)
global_mobility = pd.read_csv("data/Global_Mobility_Report.csv")
global_mobility = global_mobility[global_mobility["country_region"] == country_mobility]
global_mobility_whole = global_mobility[global_mobility["sub_region_1"].isna()]
global_mobility_whole = global_mobility_whole.set_index(global_mobility_whole["date"])
date = [int(datetime.strptime(str(x)[:10], '%Y-%m-%d').timetuple().tm_yday) for x in global_mobility_whole["date"]]

## Keep mobility only from 1st March to 31st August
dynamics_start_day = int(datetime.strptime(str('2020-03-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_start_day = date.index(dynamics_start_day)
dynamics_end_day = int(datetime.strptime(str('2020-09-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_end_day = date.index(dynamics_end_day)
final_day = date.index(date[-1]) + 1

dates_to_remove = np.hstack((np.arange(0, index_dynamics_start_day), np.arange(index_dynamics_end_day, final_day)))

## indices of the lockdown day
index_start_lockdown_day = date.index(start_date_lockdown)

mobility_data_residential_raw = global_mobility_whole["residential_percent_change_from_baseline"].to_frame().transpose()
mobility_data_workplaces_raw = global_mobility_whole["workplaces_percent_change_from_baseline"].to_frame().transpose()

# data constituting the "other" category
mobility_data_parks_raw = global_mobility_whole["parks_percent_change_from_baseline"].to_frame().transpose()
mobility_data_retail_and_recreation_raw = global_mobility_whole[
    "retail_and_recreation_percent_change_from_baseline"].to_frame().transpose()
mobility_data_transit_stations_raw = global_mobility_whole[
    "transit_stations_percent_change_from_baseline"].to_frame().transpose()
mobility_data_grocery_and_pharmacy_raw = global_mobility_whole[
    "grocery_and_pharmacy_percent_change_from_baseline"].to_frame().transpose()

from scipy.signal import savgol_filter


def transform_alpha_df(df, window):
    zeros = pd.DataFrame(data=np.zeros((1, index_start_lockdown_day)), columns=df.columns[:index_start_lockdown_day])
    data = savgol_filter(df.to_numpy().reshape(-1), window, 1)
    y = pd.DataFrame(data.reshape(1, -1), columns=df.columns).iloc[:, index_start_lockdown_day:]
    return pd.concat((zeros, y), axis=1)


mobility_data_residential = transform_alpha_df(mobility_data_residential_raw, 15)
mobility_data_workplaces = transform_alpha_df(mobility_data_workplaces_raw, 13)
mobility_data_parks = transform_alpha_df(mobility_data_parks_raw, 11)
mobility_data_retail_and_recreation = transform_alpha_df(mobility_data_retail_and_recreation_raw, 11)
mobility_data_transit_stations = transform_alpha_df(mobility_data_transit_stations_raw, 11)
mobility_data_grocery_and_pharmacy = transform_alpha_df(mobility_data_grocery_and_pharmacy_raw, 11)

# Now transform that into the alpha multipliers; in order to form alpha_other, we combine data from the last 4 categories
# above, and assume contacts in park matter for 10%, while the others matter for 30% each.

mobility_home_raw = 1 + mobility_data_residential_raw / 100.0
mobility_work_raw = 1 + mobility_data_workplaces_raw / 100.0
mobility_parks_raw = 1 + mobility_data_parks_raw / 100.0
mobility_retail_and_recreation_raw = 1 + mobility_data_retail_and_recreation_raw / 100.0
mobility_transit_stations_raw = 1 + mobility_data_transit_stations_raw / 100.0
mobility_grocery_and_pharmacy_raw = 1 + mobility_data_grocery_and_pharmacy_raw / 100.0

alpha_home = 1 + mobility_data_residential / 100.0
alpha_work = 1 + mobility_data_workplaces / 100.0
alpha_parks = 1 + mobility_data_parks / 100.0
alpha_retail_and_recreation = 1 + mobility_data_retail_and_recreation / 100.0
alpha_transit_stations = 1 + mobility_data_transit_stations / 100.0
alpha_grocery_and_pharmacy = 1 + mobility_data_grocery_and_pharmacy / 100.0

alpha_other = 0.1 * alpha_parks + 0.3 * alpha_retail_and_recreation + 0.3 * alpha_transit_stations + 0.3 * alpha_grocery_and_pharmacy

# For the schools: we reduce to alpha=0.1 for days from the start of lockdown.
data = np.ones((1, alpha_other.shape[1]))
data[:, index_start_lockdown_day:] = 0.1
alpha_school = pd.DataFrame(data=data, columns=alpha_other.columns)


# Transform to numpy and add final steady states (more days with same value as the last day, up to the day for which we
# have observed data

def df_alphas_to_np(df, extra_number_days):
    array = np.zeros(df.shape[1] + extra_number_days)
    array[:df.shape[1]] = df
    array[df.shape[1]:] = df.iloc[0, -1]
    return array


alpha_home_np = df_alphas_to_np(alpha_home, 0)
alpha_work_np = df_alphas_to_np(alpha_work, 0)
alpha_other_np = df_alphas_to_np(alpha_other, 0)
alpha_school_np = df_alphas_to_np(alpha_school, 0)

# Dynamics starts on 1st March (Remove data before 1st March) and until 31st Aug
alpha_home_np = np.delete(alpha_home_np, dates_to_remove)
alpha_work_np = np.delete(alpha_work_np, dates_to_remove)
alpha_other_np = np.delete(alpha_other_np, dates_to_remove)
alpha_school_np = np.delete(alpha_school_np, dates_to_remove)
# Dynamics starts on 1st March (Remove data before 1st March)

# np.save('data/' + subfolder + '/mobility_date', list(range(min(date), max(Observed_Data[0, :]) + 1)))
np.save('data/' + subfolder + '/mobility_home', alpha_home_np)
np.save('data/' + subfolder + '/mobility_work', alpha_work_np)
np.save('data/' + subfolder + '/mobility_other', alpha_other_np)
np.save('data/' + subfolder + '/mobility_school', alpha_school_np)
