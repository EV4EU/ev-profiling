# Import libraries
import copy
import numpy as np

# Import EVProfiler classes
from evprofiler.aggregator import Aggregator


# Define an Aggregator_Weekend class, that will aggregate the trips and EVs. Has the following parameters:
# - EVs
# - Number of EVs
class Weekend(Aggregator):
    def __init__(self, profiles, evs, n_evs, avg_speed_short, avg_speed_medium, avg_speed_long, medium_trip_min,
                 medium_trip_max, trip_length_variation, trip_start_variation, avg_speed_variation,
                 time_resolution, simulation_cycles, show_progress, initial_soc=0.85, initial_date=None):
        super().__init__(profiles, evs,
                         n_evs, time_resolution,
                         avg_speed_short, avg_speed_medium, avg_speed_long,
                         medium_trip_min, medium_trip_max, trip_length_variation, trip_start_variation,
                         avg_speed_variation,
                         simulation_cycles, initial_date,
                         show_progress)

        # Create deep copies of EVs to ensure independence
        self.evs_from_aggregator = copy.deepcopy(self.evs)

        # Access the last 'soc' value from the Aggregator for each EV
        last_soc_aggregator_weekend = [ev.soc[-1] for ev in self.evs_from_aggregator]

        # Set the 'initial_soc' for Aggregator_Weekend using the last 'soc' values
        self.initial_soc = last_soc_aggregator_weekend

    # Create the population of DrivableEVs
    def create_evs(self, n_evs):
        self.number_of_evs = n_evs

        if self.show_progress:
            print('Creating EVs...')

        self.population = self.evs_from_aggregator[:n_evs]
        for idx in range(len(self.population)):
            temp_ev = self.population[idx]
            temp_soc = self.initial_soc[idx]
            temp_ev.reset(temp_soc)
            temp_ev.set_date(self.initial_date)

    def generate_trips_distances(self):
        mean = 64.00
        std = 27.00
        trip_lengths = []

        while len(trip_lengths) < self.number_of_evs:
            value = np.round(np.random.normal(mean, std), 2)
            if value > 0:
                trip_lengths.append(value)

        return trip_lengths
