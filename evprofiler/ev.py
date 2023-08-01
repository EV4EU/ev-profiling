# Author: ECGomes
# Date: 2023-04-04

# Import libraries
import datetime
import numpy as np

from .trip import *


# Create an EV class with the following attributes:
# - Brand
# - Model
# - Battery type
# - Battery size
# - Charging Efficiency
# - Discharging Efficiency
# - Energy per km
# - SOC min
# - SOC max
# - Probability in population - this is the probability that a car of this type will be chosen from the population

class EV:
    def __init__(self, brand, model, battery_type, battery_size, charging_efficiency, discharging_efficiency,
                 energy_per_km, soc_min, soc_max, probability_in_population):
        self.brand = brand
        self.model = model
        self.battery_type = battery_type
        self.battery_size = battery_size
        self.charging_efficiency = charging_efficiency if charging_efficiency <= 1 or charging_efficiency >= 0 else 0.9
        self.discharging_efficiency = discharging_efficiency \
            if discharging_efficiency <= 1 or discharging_efficiency >= 0 else 0.9
        self.energy_per_km = energy_per_km
        self.soc_min = soc_min if soc_min >= 0.0 else 0.0
        self.soc_max = soc_max if soc_max <= 1.0 else 1.0
        self.probability_in_population = probability_in_population if probability_in_population <= 1.0 else 1.0


# Extend the class to create drivable EVs
class DrivableEV(EV):

    # __init__ method is inherited from EV class, extended with the state of charge. Creates the following attributes:
    # - State of Charge
    # - List of trips
    # - Current trip
    def __init__(self, brand, model, battery_type, battery_size, charging_efficiency, discharging_efficiency,
                 energy_per_km, soc_min, soc_max, probability_in_population, initial_soc=None, stopping_soc=None):
        super().__init__(brand, model, battery_type, battery_size, charging_efficiency, discharging_efficiency,
                         energy_per_km, soc_min, soc_max, probability_in_population)

        # EV ID
        self.ev_id = secrets.token_hex(16)

        # SOC variables
        self.soc = [initial_soc]
        self.current_soc = initial_soc
        # Attribute a random SOC to stop the EV if none is provided
        self.stopping_soc = stopping_soc if stopping_soc is not None else np.random.uniform(0.2, 0.3, 1)[0]

        # List of trips and current trip
        self.trips = []
        self.split_trips = []

        # User profile
        self.user_profile = None

        # Placeholder for the charging station pools that the EV will use
        self.default_charging_stations_pool = None
        self.segment_charging_stations_pool = None

        # Placeholder for the charging station that the EV will use
        self.default_charging_station = None
        self.segment_charging_station = None

        # Create a list of charging stations
        self.charging_stations_list = []

        # Number of stops
        self.number_of_stops = 0

        # Charging times
        self.charging_start_time = []
        self.charging_end_time = []
        self.charging_type = []
        self.charging_power = []
        self.charging_start_soc = []
        self.charging_end_soc = []

        # Driving Logs
        self.driving_start_time = []
        self.driving_end_time = []
        self.driving_start_soc = []
        self.driving_end_soc = []
        self.driving_distance = []
        self.driving_energy_consumption = []

        # Flexibility logs
        self.flexibility_start_time = []
        self.flexibility_end_time = []
        self.flexibility_type = []
        self.flexibility_power = []
        self.flexibility_start_soc = []
        self.flexibility_end_soc = []

    # Format the print
    def __repr__(self):
        return f"{self.brand} {self.model} with {self.battery_size} kWh battery"

    # Create a method to add a trip to the EV
    def add_trip(self, trip):
        self.trips.append(trip)

    # Create a method to charge the EV
    def charge(self, charge_amount):
        self.current_soc += charge_amount * self.charging_efficiency

        if self.current_soc > 1.0:
            self.current_soc = 1.0

        self.soc.append(self.current_soc)

    # Create a method to charge the EV to a certain SOC.
    # This is used for the segment charging and outputs the amount of energy charged and time taken to charge
    def charge_to_soc(self, target_soc, cs_charging_power, cs_charging_efficiency, charge_ev=True):
        # Calculate the amount of energy needed to charge the EV
        energy_needed = (target_soc - self.current_soc) * self.battery_size if target_soc > self.current_soc else 0.0

        cs_charging_efficiency = min(cs_charging_efficiency, self.charging_efficiency)

        # Calculate the time needed to charge the EV
        time_needed = energy_needed / (cs_charging_power * cs_charging_efficiency)
        time_needed = datetime.timedelta(hours=time_needed)

        # Charge the EV
        if charge_ev:
            self.charge(energy_needed / self.battery_size)

        return energy_needed, time_needed

    # Create a method to log the charging of the EV
    def log_charging(self, charging_start_time, charging_end_time, charging_type, charging_power, charging_start_soc,
                     charging_end_soc):
        self.charging_start_time.append(charging_start_time)
        self.charging_end_time.append(charging_end_time)
        self.charging_type.append(charging_type)
        self.charging_power.append(charging_power)
        self.charging_start_soc.append(charging_start_soc)
        self.charging_end_soc.append(charging_end_soc)

    # Create a method to log the flexibility of charge of the EV
    def log_flexibility(self, flexibility_start_time, flexibility_end_time, flexibility_type, flexibility_power,
                        flexibility_start_soc, flexibility_end_soc):
        self.flexibility_start_time.append(flexibility_start_time)
        self.flexibility_end_time.append(flexibility_end_time)
        self.flexibility_type.append(flexibility_type)
        self.flexibility_power.append(flexibility_power)
        self.flexibility_start_soc.append(flexibility_start_soc)
        self.flexibility_end_soc.append(flexibility_end_soc)

    # Create a method to log the driving of the EV
    def log_driving(self, driving_start_time, driving_end_time, driving_start_soc, driving_end_soc, driving_distance,
                    driving_energy_consumption):
        self.driving_start_time.append(driving_start_time)
        self.driving_end_time.append(driving_end_time)
        self.driving_start_soc.append(driving_start_soc)
        self.driving_end_soc.append(driving_end_soc)
        self.driving_distance.append(driving_distance)
        self.driving_energy_consumption.append(driving_energy_consumption)

    # Reset the SoC
    def reset_initial_soc(self, initial_soc):
        self.current_soc = initial_soc
        self.soc = [initial_soc]

    #  Reset logs
    def reset_logs(self):
        # Charging times
        self.charging_start_time = []
        self.charging_end_time = []
        self.charging_type = []
        self.charging_power = []
        self.charging_start_soc = []
        self.charging_end_soc = []

        # Driving Logs
        self.driving_start_time = []
        self.driving_end_time = []
        self.driving_start_soc = []
        self.driving_end_soc = []
        self.driving_distance = []
        self.driving_energy_consumption = []

        # Flexibility logs
        self.flexibility_start_time = []
        self.flexibility_end_time = []
        self.flexibility_type = []
        self.flexibility_power = []
        self.flexibility_start_soc = []
        self.flexibility_end_soc = []

    # Reset the stopping points
    def reset_stops(self):
        self.number_of_stops = 0

    # Reset the EV for a new day
    def reset(self, initial_soc):
        self.reset_initial_soc(initial_soc)
        self.reset_logs()
        self.reset_stops()

    # Create a method to discharge the EV
    def discharge(self, discharge_amount):
        self.current_soc -= discharge_amount / self.discharging_efficiency

        if self.current_soc < 0:
            self.current_soc = 0

        self.soc.append(self.current_soc)

    # Create a method to assign the allowed charging stations by the profile
    def assign_charging_stations(self):

        # Check if charging stations were already assigned
        if len(self.charging_stations_list) > 0:
            return

        # Get the charging stations from the profile
        allowed_charging_stations = self.trips[0].profile.allowed_charging_stations

        # Default stations
        self.default_charging_stations_pool = allowed_charging_stations['Default']

        # Segment stations
        self.segment_charging_stations_pool = allowed_charging_stations['Segment']

        # Pick the default charging station
        self.default_charging_station = np.random.choice(self.default_charging_stations_pool, 1,
                                                         p=[x.charging_station_probability for x in
                                                            self.default_charging_stations_pool])[0]
        self.default_charging_station.ev_id = self.ev_id

        # Pick the segment charging station
        self.segment_charging_station = np.random.choice(self.segment_charging_stations_pool, 1,
                                                         p=[x.charging_station_probability for x in
                                                            self.segment_charging_stations_pool])[0]
        self.segment_charging_station.ev_id = self.ev_id

        # Create a list of charging stations
        self.charging_stations_list = [self.default_charging_station, self.segment_charging_station]

    # Create a method to check necessity of trip splitting and charging
    def trip_splitting(self):

        # Placeholder for the list of trips
        list_split_trips = []

        # Placeholder for the current time
        current_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

        # Check if the profile allows charging during the day
        charge_during_day = self.user_profile.charge_during_day
        charge_during_night = self.user_profile.charge_during_night
        soc_min_tocharge = self.user_profile.soc_min_tocharge

        # Check if SOC is below the minimum. If yes, charge the EV to capacity specified on the Profile.
        if self.current_soc < self.stopping_soc:
            temp_soc = self.current_soc

            charge_energy, charge_time = self.charge_to_soc(soc_min_tocharge,
                                                            self.default_charging_station.charging_station_power,
                                                            self.default_charging_station.charging_station_efficiency)
            self.log_charging(current_time, current_time + charge_time,
                              self.default_charging_station.charging_station_type,
                              self.default_charging_station.charging_station_power,
                              temp_soc,
                              self.current_soc)

            # Update current time
            current_time += charge_time

        # Loop over the trips
        for i in range(len(self.trips)):

            trip = self.trips[i]

            # Check how much the EV can cover with the current SOC
            possible_trip_length = np.round(
                (self.current_soc - self.stopping_soc) * self.battery_size / self.energy_per_km, 2)

            # If the trip can be covered, add it to the list
            if possible_trip_length >= trip.trip_length:
                list_split_trips.append(trip)

                # SOC before the trip
                temp_soc = self.current_soc

                # Discharge the required SOC
                self.discharge(trip.trip_required_soc)

                # Register driving logs
                self.log_driving(trip.trip_start_time, trip.trip_end_time,
                                 temp_soc, self.current_soc, trip.trip_length, trip.trip_required_soc)

                # Update the current time
                current_time = trip.trip_end_time

                trip.covered_distance = trip.trip_length

            # If the trip cannot be covered, split it
            else:

                current_covered_distance = 0
                current_trip_start_time = trip.trip_start_time

                # Loop over the trip
                while current_covered_distance < trip.trip_length:

                    # Calculate the possible trip length
                    possible_trip_length = np.round(
                        (self.current_soc - self.stopping_soc) * self.battery_size / self.energy_per_km, 2)

                    # Calculate the trip length
                    trip_length = min(possible_trip_length, trip.trip_length - current_covered_distance)

                    # Calculate the trip duration
                    trip_duration = datetime.timedelta(hours=trip_length / trip.trip_speed)

                    # Calculate the trip end time
                    trip_end_time = current_trip_start_time + trip_duration

                    # Create a new trip
                    new_trip = DrivableTrip(trip_length, trip.trip_speed,
                                            current_trip_start_time, trip_end_time,
                                            trip_duration, profile=trip.profile, trip_id=trip.trip_id)
                    new_trip.ev = self
                    new_trip.ev_id = self.ev_id
                    new_trip.trip_energy_consumption = trip_length * self.energy_per_km
                    new_trip.trip_required_soc = new_trip.trip_energy_consumption / self.battery_size
                    new_trip.assign_ev_battery_size()
                    new_trip.covered_trip_length = trip_length

                    # Add the trip to the list
                    list_split_trips.append(new_trip)

                    # SOC before the trip
                    temp_soc = self.current_soc

                    # Discharge the required SOC
                    self.discharge(new_trip.trip_required_soc)

                    # Register driving logs
                    self.log_driving(new_trip.trip_start_time, new_trip.trip_end_time,
                                     temp_soc, self.current_soc, new_trip.trip_length, new_trip.trip_required_soc)

                    # Update the current time
                    current_time = new_trip.trip_end_time

                    # Update the current covered distance
                    current_covered_distance += trip_length

                    # Charge the EV using a segment charging station
                    charge_energy, \
                        charge_time = self.charge_to_soc(soc_min_tocharge,
                                                         self.segment_charging_station.charging_station_power,
                                                         self.segment_charging_station.charging_station_efficiency)

                    # Register charging logs
                    self.log_charging(current_time, current_time + charge_time,
                                      self.segment_charging_station.charging_station_type,
                                      self.segment_charging_station.charging_station_power,
                                      temp_soc,
                                      self.current_soc)

                    # Update the current time
                    current_time += charge_time

                    # Update the current trip start time
                    current_trip_start_time = current_time

                    # Break the loop if the trip is the last one
                    if current_covered_distance >= trip.trip_length:
                        break

            # Check if there are more trips
            if i == len(self.trips) - 1:
                break

            # Check if the next trip is in the same day as this one
            if self.trips[i + 1].trip_start_time.day == trip.trip_start_time.day:
                # Check if EVs can charge during day
                if charge_during_day & (self.current_soc <= soc_min_tocharge):
                    # Log the flexibility
                    self.log_flexibility(current_time, self.trips[i + 1].trip_start_time,
                                         self.default_charging_station.charging_station_type,
                                         self.default_charging_station.charging_station_power,
                                         0.0,
                                         0.0)

                    # Get the current soc
                    temp_soc = self.current_soc

                    # Charge the EV using a segment charging station
                    charge_energy, \
                        charge_time = self.charge_to_soc(soc_min_tocharge,
                                                         self.default_charging_station.charging_station_power,
                                                         self.default_charging_station.charging_station_efficiency)

                    # Register charging logs
                    self.log_charging(current_time, current_time + charge_time,
                                      self.default_charging_station.charging_station_type,
                                      self.default_charging_station.charging_station_power,
                                      temp_soc,
                                      self.current_soc)

                    # Update the current time
                    current_time += charge_time

            # If the next trip is in the next day, charge the EV using a default charging station
            else:
                # Check if EVs can charge during night
                if charge_during_night & (self.current_soc <= soc_min_tocharge):
                    # print('during night')

                    # Log the flexibility
                    self.log_flexibility(current_time, self.trips[i + 1].trip_start_time,
                                         self.default_charging_station.charging_station_type,
                                         self.default_charging_station.charging_station_power,
                                         0.0,
                                         0.0)

                    # Get the current soc
                    temp_soc = self.current_soc

                    # Charge the EV using a default charging station
                    charge_energy, \
                        charge_time = self.charge_to_soc(soc_min_tocharge,
                                                         self.default_charging_station.charging_station_power,
                                                         self.default_charging_station.charging_station_efficiency)

                    # Register charging logs
                    self.log_charging(current_time, current_time + charge_time,
                                      self.default_charging_station.charging_station_type,
                                      self.default_charging_station.charging_station_power,
                                      temp_soc,
                                      self.current_soc)

                    # Update the current time
                    current_time += charge_time

        # Update the trips
        self.split_trips = list_split_trips

    # Assign a profile if none is assigned
    def assign_profile(self, profile):
        # Check if the profile is already assigned
        if self.user_profile is None:
            # Assign the profile
            self.user_profile = profile
