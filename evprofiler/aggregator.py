# Author: ECGomes
# Date: 2023-04-09

# Import libraries
import numpy as np
import pandas as pd
import secrets
import datetime
import tqdm.notebook as tqdm

# Import EVProfiler classes
from .ev import *
from .profile import *
from .station import *
from .trip import *


# Define an Aggregator class, that will aggregate the trips and EVs. Has the following parameters:
# - EVs
# - Number of EVs
class Aggregator:
    def __init__(self, profiles, evs,
                 n_evs=1, time_resolution=1,
                 avg_speed_short=60, avg_speed_medium=80, avg_speed_long=100,
                 medium_trip_min=50, medium_trip_max=100, trip_length_variation=0.05, trip_start_variation=1.0,
                 avg_speed_variation=1.0,
                 simulation_cycles=1, initial_date=None,
                 show_progress=True):
        # Profile assign
        self.profiles = profiles

        # Temporal resolution
        self.time_resolution = time_resolution

        # Simulation cycle definition
        self.simulation_cycles = simulation_cycles

        # Assign the EVs to the aggregator
        self.evs = evs

        # Number of EVs of the simulation
        self.number_of_evs = n_evs

        # Initial date of the simulation
        self.initial_date = initial_date if initial_date is not None else datetime.datetime(2023, 1, 1, 0, 0, 0)

        # Create a list of trips
        self.trips = None
        self.segments = []
        self.population = None
        self.assigned_evs = []

        # Trip creation parameters
        self.avg_speed_short = avg_speed_short
        self.avg_speed_medium = avg_speed_medium
        self.avg_speed_long = avg_speed_long
        self.medium_trip_min = medium_trip_min
        self.medium_trip_max = medium_trip_max
        self.trip_length_variation = trip_length_variation
        self.trip_start_variation = trip_start_variation
        self.avg_speed_variation = avg_speed_variation

        # Placeholder for simulation results
        self.simulation_variables = {}
        self.simulation_dataframes = {}
        self.simulation_step = 1

        # Verbose flags
        self.show_progress = show_progress
        self.show_bars = np.logical_not(show_progress)

    # Add an EV to the fleet
    def add_ev(self, ev):
        self.evs.append(ev)

    # Assign a trip to an EV
    def assign_trip(self, ev, trip):
        ev.add_trip(trip)

    # Create the population of DrivableEVs according to NumPy's random.choice
    def create_evs(self, n_evs):
        # Assign the number of EVs to the aggregator
        self.number_of_evs = n_evs

        # Print the progress
        if self.show_progress:
            print('Creating EVs...')

        # Create an EV population according to the given probabilities of the EVs
        population_choice = np.random.choice(self.evs, n_evs,
                                             p=[ev.probability_in_population for ev in self.evs])

        # Create a list of DrivableEVs
        self.population = np.array([DrivableEV(ev.brand, ev.model, ev.battery_type, ev.battery_size,
                                               ev.charging_efficiency, ev.discharging_efficiency,
                                               ev.energy_per_km, ev.soc_min, ev.soc_max,
                                               ev.probability_in_population,
                                               initial_soc=np.round(np.random.uniform(ev.soc_min,
                                                                                      ev.soc_max, 1)[0], 2))
                                    for ev in tqdm.tqdm(population_choice, disable=self.show_bars)])

    # Create trips for a day based on the number of EVs
    def generate_trips_distances(self):

        # Based on a Gamma distribution
        shape, scale = 46. * .01, 46. * 2
        trip_lengths = np.round(np.random.gamma(shape, scale, self.number_of_evs), 2)

        return trip_lengths

    # Get a user profile based on the trip type
    def get_user_profile(self, trip_type):
        user_profiles = [profile for profile in self.profiles if profile.trip_type == trip_type]
        user_profile = np.random.choice(user_profiles, 1)[0]
        return user_profile

    # Trips for all the simulated days
    def create_all_trips(self):

        # Show progress
        if self.show_progress:
            print('Creating trips...')

        # Placeholder for all the trips
        self.trips = []

        # Get the generated trips for a day
        trip_lengths = self.generate_trips_distances()

        # Create the initial trip objects
        self.trips = [Trip(trip_length, None, None, None) for trip_length in trip_lengths]
        return

    # Create the drivable trips for the EV based on the profile
    def create_drivable_trip(self, ev, trip, day, user_profile):

        # Calculate trip type based on distance
        trip_type = 'Short' if trip.trip_length <= self.medium_trip_min else 'Medium' if self.medium_trip_min < trip.trip_length <= self.medium_trip_max else 'Long'

        # Get the trip length
        trip_length = trip.trip_length

        # Add some variation to the trip length
        trip_length = np.round(trip_length * np.random.uniform(1 - self.trip_length_variation,
                                                               1 + self.trip_length_variation, 1)[0], 2)

        # Set the trip length while respecting the type
        if trip_type == 'Short':
            if trip_length > self.medium_trip_min:
                trip_length = self.medium_trip_min
        elif trip_type == 'Medium':
            if trip_length < self.medium_trip_min:
                trip_length = self.medium_trip_min
            elif trip_length > self.medium_trip_max:
                trip_length = self.medium_trip_max
        elif trip_type == 'Long':
            if trip_length < self.medium_trip_max:
                trip_length = self.medium_trip_max

        # Check how many trips the user has
        user_trips = len(user_profile.profile_trip_schedule.schedule_start)

        # Calculate average speed based on trip type
        avg_speed = self.avg_speed_short if trip_type == 'Short' else self.avg_speed_medium \
            if trip_type == 'Medium' else self.avg_speed_long

        # Calculate the required socs for trip segment
        for i in range(user_trips):
            # Get a random start centered around schedule start
            segment_start = \
            np.random.uniform(user_profile.profile_trip_schedule.schedule_start[i] - self.trip_start_variation,
                              user_profile.profile_trip_schedule.schedule_start[i] + self.trip_start_variation, 1)[0]

            # Segment start in datetime
            segment_start = datetime.datetime(self.initial_date.year, self.initial_date.month, self.initial_date.day,
                                              int(segment_start), int((segment_start - int(segment_start)) * 60))
            segment_start = segment_start + datetime.timedelta(days=day)

            # Get a random average speed centered on avg_speed with variation
            segment_avg_speed = np.random.uniform(avg_speed - self.avg_speed_variation,
                                                  avg_speed + self.avg_speed_variation, 1)[0]

            # Duration is the trip length divided by the average speed
            duration = trip_length / user_trips / segment_avg_speed \
                if segment_avg_speed > 0 else trip_length / avg_speed
            duration = datetime.timedelta(hours=duration)

            # Segment end in datetime
            segment_end = segment_start + duration

            # Create the drivable trip
            drivable_trip = DrivableTrip(trip_length / user_trips,
                                         segment_avg_speed,
                                         segment_start, segment_end,
                                         duration, profile=user_profile)

            # Establish the required parameters for the trip
            drivable_trip.ev = ev
            drivable_trip.ev_id = ev.ev_id
            drivable_trip.calculate_trip_energy_consumption()
            drivable_trip.calculate_trip_required_soc()
            drivable_trip.assign_ev_battery_size()

            # Append the drivable trip to the EV
            ev.add_trip(drivable_trip)
            ev.assign_charging_stations()

        return

    # Assign all the trips for all the days
    def assign_all_trips(self):

        # Show progress
        if self.show_progress:
            print('Assigning trips to EVs...')

        # Assigned EVs
        ev_ids = [ev.ev_id for ev in self.population]

        # Get the sum of the battery sizes
        battery_sizes = np.array([ev.battery_size for ev in self.population])
        battery_sizes_unique = np.unique(battery_sizes)

        # Sum of unique battery sizes
        battery_sizes_unique_sum = sum(battery_sizes_unique)

        # Create a dictionary with the battery sizes and their index in the population
        battery_sizes_dict = {}
        for i in range(len(battery_sizes_unique)):
            battery_sizes_dict['B{}'.format(battery_sizes_unique[i])] = [ev for ev in self.population if
                                                                         ev.battery_size == battery_sizes_unique[i]]

        # Create a list with the probabilities of the EVs to get a trip
        probabilities = battery_sizes_unique / battery_sizes_unique_sum

        # Sort the segments by trip length and assign it to a temporary variable
        sorted_trips = sorted(self.trips, key=lambda x: x.trip_length, reverse=True)
        # print([trip.trip_length for trip in sorted_trips])
        # trip_type = ['Short' if trip.trip_length <= self.medium_trip_min else 'Medium'
        # if self.medium_trip_min < trip.trip_length <= self.medium_trip_max else 'Long' for trip in sorted_trips]
        # user_profiles = [self.get_user_profile(trip_type) for trip_type in trip_type]

        # Created profiles
        user_profiles = []
        for profile in self.profiles:
            # Append the profile to the created profiles, based on the probability and the number of EVs
            user_profiles += [profile] * int(np.round(profile.profile_probability * self.number_of_evs))

        # Sort the profiles by trip length
        user_profiles = sorted(user_profiles, key=lambda x: x.trip_type, reverse=False)

        for idx in tqdm.tqdm(range(len(sorted_trips)), disable=self.show_bars):
            # for idx in range(len(sorted_trips)):
            trip = sorted_trips[idx]
            user_profile = user_profiles[idx]

            # Choose an EV from the population based on the battery size
            chosen_battery_size = np.random.choice(battery_sizes_unique, 1, p=probabilities)[0]

            # Get a random EV from the population with the chosen battery size
            choice_ev = np.random.choice(battery_sizes_dict['B{}'.format(chosen_battery_size)], 1)[0]

            # Get the respective EV from the assigned EVs
            ev = self.population[ev_ids.index(choice_ev.ev_id)]

            # Assign the profile to the EV
            ev.assign_profile(user_profile)

            for day in range(self.simulation_cycles):
                # Create the drivable trip
                self.create_drivable_trip(ev, trip, day, user_profile)

            # Remove the EV from the population
            battery_sizes_dict['B{}'.format(chosen_battery_size)].remove(choice_ev)

            # Update the probabilities
            if len(battery_sizes_dict['B{}'.format(chosen_battery_size)]) == 0:
                battery_sizes_unique = np.delete(battery_sizes_unique,
                                                 np.where(battery_sizes_unique == chosen_battery_size)[0][0])
                battery_sizes_unique_sum = sum(battery_sizes_unique)
                probabilities = battery_sizes_unique / battery_sizes_unique_sum

        return

    # Split the EV trips
    def split_all_trips(self):

        if self.show_progress:
            print('Splitting trips...')

        for ev in tqdm.tqdm(self.population, disable=self.show_bars):
            ev.trip_splitting()

    def run_sim(self):
        # If the simulation's number of days is less than 1, set to 1 and run the first day
        if self.simulation_cycles < 1:
            print('Simulation days must be greater than 0. Setting to 1.')
            self.simulation_cycles = 1

        # Print the simulation progress
        if self.show_progress:
            print('Simulating  {:003d} cycles...'.format(self.simulation_cycles))

        # Run the simulation
        self.create_evs(self.number_of_evs)
        self.create_all_trips()
        self.assign_all_trips()
        self.split_all_trips()

        self.simulation_dataframes = self.get_dataframes()

    # Create a DataFrame of the population
    def population_dataframe(self):
        # Create a DataFrame of the population
        population_dataframe = pd.DataFrame([ev.__dict__ for ev in self.population])

        # Return the DataFrame
        return population_dataframe

    # Create a DataFrame of the trips
    def assigned_trips_dataframe(self):
        # Create a DataFrame of the trips
        trips_dataframe = pd.DataFrame([trip.trip_length for trip in self.trips])
        trips_dataframe.columns = ["trip_length"]

        # Return the DataFrame
        return trips_dataframe

    # Create a DataFrame of assigned Trips and all the information in the list
    def assigned_segments_dataframe(self):
        # Create a DataFrame of the population
        assigned_trips_dataframe = pd.DataFrame([trip.__dict__ for ev in self.population for trip in ev.trips])

        # Return the DataFrame
        return assigned_trips_dataframe

    # Create a DataFrame of the EVs' split trips
    def assigned_split_trips_dataframe(self):
        # Create a DataFrame of the population
        assigned_split_trips_dataframe = pd.DataFrame(
            [trip.__dict__ for ev in self.population for trip in ev.split_trips])

        # Return the DataFrame
        return assigned_split_trips_dataframe

    # Create a DataFrame of the charging stations used by the EVs
    def assigned_cs_dataframe(self):
        # Create a DataFrame of the population
        assigned_cs_dataframe = pd.DataFrame(
            [cs.__dict__ for ev in self.assigned_evs for cs in ev.charging_stations_list])

        # Return the DataFrame
        return assigned_cs_dataframe

    # Create a DataFrame composed by:
    # EV charging start time, EV charging end time, EV charging start soc, EV charging end soc, EV ID
    def cs_charging_dataframe(self):

        # Create placeholder lists
        start_times = []
        end_times = []

        station_type = []
        station_power = []

        start_socs = []
        end_socs = []

        ev_ids = []

        # Iterate through the EVs
        for ev in self.population:

            # Check if there were any events
            if len(ev.charging_start_time) == 0:
                continue

            # Iterate through the events
            for i in range(len(ev.charging_start_time)):
                start_times.append(ev.charging_start_time[i])
                end_times.append(ev.charging_end_time[i])

                station_type.append(ev.charging_type[i])
                station_power.append(ev.charging_power[i])

                start_socs.append(ev.charging_start_soc[i])
                end_socs.append(ev.charging_end_soc[i])

                ev_ids.append(ev.ev_id)

        # Create the DataFrame
        charging_dataframe = pd.DataFrame({"start_time": start_times, "end_time": end_times, "start_soc": start_socs,
                                           "end_soc": end_socs, "ev_id": ev_ids, "station_type": station_type,
                                           "station_power": station_power})

        # Return the DataFrame
        return charging_dataframe

    # Create a DataFrame of EV charging history
    def ev_charging_history_dataframe(self):

        # Create a date range with a frequency of 1 hour
        date_range = pd.date_range(start=self.initial_date, periods=self.simulation_cycles * 24, freq='H')

        # Create the index for the dataframe
        cs_df = self.cs_charging_dataframe()
        index = cs_df['ev_id'].unique()

        # Create the dataframe
        ev_charging_history = pd.DataFrame(index=index, columns=date_range)
        ev_charging_type = pd.DataFrame(index=index, columns=date_range)

        if self.show_progress:
            print('Creating EV charging history DataFrame...')

        # Fill the dataframe with the charging data
        for i in tqdm.tqdm(range(len(cs_df)), disable=self.show_bars):
            ev_id = cs_df['ev_id'][i]
            start = pd.to_datetime(cs_df['start_time'][i]).floor('H')
            end = pd.to_datetime(cs_df['end_time'][i]).floor('H')
            # end = cs_df['end_time'][i]
            power = cs_df['station_power'][i]
            ev_charging_history.loc[ev_id, start:end] = power
            ev_charging_type.loc[ev_id, start:end] = cs_df['station_type'][i]

        # Fill the NaN values with 0
        ev_charging_history.fillna(0.0, inplace=True)
        ev_charging_type.fillna(0, inplace=True)

        # Return the dataframe
        return ev_charging_history, ev_charging_type

    # Create a DataFrame of the EVs' driving history
    def ev_driving_dataframe(self):

        # Create placeholder lists
        start_times = []
        end_times = []

        distance = []

        ev_ids = []

        # Iterate through the EVs
        for ev in self.population:

            # Check if there were any events
            if len(ev.driving_start_time) == 0:
                continue

            # Iterate through the events
            for i in range(len(ev.driving_start_time)):
                start_times.append(ev.driving_start_time[i])
                end_times.append(ev.driving_end_time[i])

                distance.append(ev.driving_distance[i])

                ev_ids.append(ev.ev_id)

        # Create the DataFrame
        driving_dataframe = pd.DataFrame({"start_time": start_times, "end_time": end_times, "distance": distance,
                                          "ev_id": ev_ids})

        # Return the DataFrame
        return driving_dataframe

    # Create a DataFrame of EV driving history
    def ev_driving_history_dataframe(self):

        # Create a date range with a frequency of 1 hour
        date_range = pd.date_range(start=self.initial_date, periods=self.simulation_cycles * 24, freq='H')

        # Create the index for the dataframe
        drv_df = self.ev_driving_dataframe()
        index = drv_df['ev_id'].unique()

        # Create the dataframe
        ev_driving_history = pd.DataFrame(index=index, columns=date_range)

        if self.show_progress:
            print('Creating EV driving history DataFrame...')

        # Fill the dataframe with the charging data
        for i in tqdm.tqdm(range(len(drv_df)), disable=self.show_bars):
            ev_id = drv_df['ev_id'][i]
            start = pd.to_datetime(drv_df['start_time'][i]).floor('H')
            end = pd.to_datetime(drv_df['end_time'][i]).floor('H')
            ev_driving_history.loc[ev_id, start:end] = 1

        # Fill the NaN values with 0
        ev_driving_history.fillna(0, inplace=True)

        # Return the dataframe
        return ev_driving_history

    # Create a DataFrame of when the EVs are not charging or driving
    def ev_stopped(self):

        # Create a date range with a frequency of 1 hour
        date_range = pd.date_range(start=self.initial_date, periods=self.simulation_cycles * 24, freq='H')

        # Create the index for the dataframe
        drv_df = self.ev_driving_dataframe()
        index = drv_df['ev_id'].unique()

        # Create the dataframe
        ev_stopped = pd.DataFrame(index=index, columns=date_range)

        if self.show_progress:
            print('Creating EV stopped DataFrame...')

        # Fill the dataframe with the driving data
        for i in tqdm.tqdm(range(len(drv_df)), disable=self.show_bars):
            ev_id = drv_df['ev_id'][i]
            start = pd.to_datetime(drv_df['start_time'][i]).floor('H')
            end = pd.to_datetime(drv_df['end_time'][i]).floor('H')
            ev_stopped.loc[ev_id, start:end] = 1

        # Reverse the dataframe for the driving information
        ev_stopped = 1 - ev_stopped

        # Do the same thing for the charging data
        cs_df = self.cs_charging_dataframe()

        for i in tqdm.tqdm(range(len(cs_df)), disable=self.show_bars):
            ev_id = cs_df['ev_id'][i]
            start = pd.to_datetime(cs_df['start_time'][i]).floor('H')
            end = pd.to_datetime(cs_df['end_time'][i]).floor('H')
            ev_stopped.loc[ev_id, start:end] = 1

        # Fill the NaN values with 0
        ev_stopped.fillna(0, inplace=True)

        # Return the dataframe
        return ev_stopped

    # Create a DataFrame of EV driving history
    def ev_flexibility_dataframe(self):

        # Create placeholder lists
        start_times = []
        end_times = []

        power = []

        ev_ids = []

        # Iterate through the EVs
        for ev in self.population:

            # Check if there were any events
            if len(ev.flexibility_start_time) == 0:
                continue

            # Iterate through the events
            for i in range(len(ev.flexibility_start_time)):
                start_times.append(ev.flexibility_start_time[i])
                end_times.append(ev.flexibility_end_time[i])

                power.append(ev.flexibility_power[i])

                ev_ids.append(ev.ev_id)

        # Create the DataFrame
        flex_dataframe = pd.DataFrame({"start_time": start_times, "end_time": end_times, "power": power,
                                       "ev_id": ev_ids})

        # Return the DataFrame
        return flex_dataframe

    # Create a DataFrame of EV flexibility
    def ev_flexibility_history_dataframe(self):

        # Create a date range with a frequency of 1 hour
        date_range = pd.date_range(start=self.initial_date, periods=self.simulation_cycles * 24, freq='H')

        # Create the index for the dataframe
        flex_df = self.ev_flexibility_dataframe()
        index = flex_df['ev_id'].unique()

        # Create the dataframe
        ev_flex = pd.DataFrame(index=index, columns=date_range)

        if self.show_progress:
            print('Creating EV flexibility DataFrame...')

        # Fill the dataframe with the driving data
        for i in tqdm.tqdm(range(len(flex_df)), disable=self.show_bars):
            ev_id = flex_df['ev_id'][i]
            start = pd.to_datetime(flex_df['start_time'][i]).floor('H')
            end = pd.to_datetime(flex_df['end_time'][i]).floor('H')
            ev_flex.loc[ev_id, start:end] = 1

        # Fill the NaN values with 0
        ev_flex.fillna(0, inplace=True)

        return ev_flex

    # Get all the DataFrames for storage
    def get_dataframes(self):

        if self.show_progress:
            print('Creating DataFrames...')

        # Population DataFrame
        population_dataframe = self.population_dataframe()

        # Assigned Trips
        assigned_trips_dataframe = self.assigned_trips_dataframe()

        # Assigned Segments
        assigned_segments_dataframe = self.assigned_segments_dataframe()

        # Assigned Split Trips
        assigned_split_trips_dataframe = self.assigned_split_trips_dataframe()

        # Assigned Charging Stations
        assigned_charging_stations_dataframe = self.assigned_cs_dataframe()

        # Get the DataFrames
        cs_history_dataframe = self.cs_charging_dataframe()
        ev_charging_history_dataframe, ev_charging_type_dataframe = self.ev_charging_history_dataframe()
        ev_driving_dataframe = self.ev_driving_dataframe()
        ev_driving_history_dataframe = self.ev_driving_history_dataframe()
        ev_stopped = self.ev_stopped()
        ev_flex = self.ev_flexibility_history_dataframe()

        ev_flex_power = ev_flex.copy(deep=True)
        for id in ev_flex_power.index:
            temp_power = self.population_dataframe().loc[self.population_dataframe()['ev_id'] == id][
                'default_charging_station'].values[0].charging_station_power
            ev_flex_power.loc[ev_flex_power.index == id] = ev_flex_power.loc[ev_flex_power.index == id] * temp_power

        # Place everything in a dictionary
        dataframes = {"population": population_dataframe,
                      "assigned_trips": assigned_trips_dataframe,
                      "assigned_segments": assigned_segments_dataframe,
                      "assigned_split_trips": assigned_split_trips_dataframe,
                      "assigned_charging_stations": assigned_charging_stations_dataframe,
                      "cs_history": cs_history_dataframe,
                      "ev_charging_history": ev_charging_history_dataframe,
                      "ev_charging_type": ev_charging_type_dataframe,
                      "ev_driving": ev_driving_dataframe,
                      "ev_driving_history": ev_driving_history_dataframe,
                      "ev_stopped": ev_stopped,
                      "ev_flexibility": ev_flex,
                      "ev_flexibility_power": ev_flex_power}

        # Return the DataFrames
        return dataframes
