import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_particle_data(filename):
    chunk_size = 10000  # Adjust based on file size and system memory
    chunks = pd.read_csv(filename, header=None, chunksize=chunk_size)
    df = pd.concat(chunks)
    df.columns = ['Timestep', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ', 'Mass']
    df['Timestep'] = df['Timestep'].astype('int32')
    df['Mass'] = df['Mass'].astype('float32')
    return df

def get_run_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

def process_file(filename):
    data = read_particle_data(filename)
    grouped = data.groupby('Timestep')

    timesteps = grouped.groups.keys()
    previous_count = len(grouped.get_group(next(iter(timesteps))))

    total_collisions = 0
    collisions_over_time = []
    average_mass_over_time = []
    total_mass_stars_over_time = []

    for timestep in list(timesteps)[1:]:
        current_group = grouped.get_group(timestep)
        current_count = len(current_group)
        collisions_this_step = previous_count - current_count
        if collisions_this_step > 0:
            total_collisions += collisions_this_step
        collisions_over_time.append(total_collisions)

        non_star_particles = current_group[current_group['Mass'] < 1e30]
        average_mass_over_time.append(non_star_particles['Mass'].mean())

        total_mass_stars_over_time.append(current_group[current_group['Mass'] > 1e30]['Mass'].sum())
        previous_count = current_count

    return collisions_over_time, average_mass_over_time, total_mass_stars_over_time


run_files = get_run_files("runs")
collisions_data = {}
average_mass_data = {}
total_mass_stars_data = {}

for filename in run_files:
    collisions, average_mass, total_mass_stars = process_file(filename)
    collisions_data[filename] = collisions
    average_mass_data[filename] = average_mass
    total_mass_stars_data[filename] = total_mass_stars

# Plotting
plt.figure(figsize=(12, 6))

# Collisions plot
plt.subplot(1, 3, 1)
for filename, collisions in collisions_data.items():
    plt.plot(collisions, label=os.path.basename(filename).replace('.csv', ''))
plt.title("Total Collisions Over Time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Timestep")
plt.ylabel("Cumulative Number of Collisions")

# Average mass plot
plt.subplot(1, 3, 2)
for filename, avg_mass in average_mass_data.items():
    plt.plot(avg_mass, label=os.path.basename(filename).replace('.csv', ''))
plt.title("Average Particle Mass Over Time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Timestep")
plt.ylabel("Average Mass")

# Total mass of stars plot
plt.subplot(1, 3, 3)
for filename, total_mass_stars in total_mass_stars_data.items():
    plt.plot(total_mass_stars, label=os.path.basename(filename).replace('.csv', ''))
plt.title("Total Mass of Stars Over Time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Timestep")
plt.ylabel("Total Mass of Stars")

plt.tight_layout()
plt.legend()

plt.savefig("inclination_graph.svg", format="svg")
plt.show()