import pygame
import pandas as pd

# Pygame setup
pygame.init()
width, height = 1400, 750
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()
AU = 149.6e9
EarthMass = 5.972e24

# Function to convert simulation coordinates to screen coordinates
def to_screen_coords(pos, scale=5*AU / height):
    coords = int(width / 2 + pos[0] / scale), int(height / 2 - pos[1] / scale)
    return coords

def read_particle_data(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, header=None)
    df.columns = ['Timestep', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ', 'Mass']
    return df

data = read_particle_data("data.csv")
unique_timesteps = data['Timestep'].unique()  # Get unique timesteps

running = True
paused = False
current_index = 0  # Index to keep track of the current timestep

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_RIGHT and paused:
                current_index = min(current_index + 1, len(unique_timesteps) - 1)
            elif event.key == pygame.K_LEFT and paused:
                current_index = max(current_index - 1, 0)

    timestep = unique_timesteps[current_index]

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the particles
    for particle in data[data['Timestep'] == timestep].values:
        particle_pos = particle[1:3]  # Only X and Y coordinates
        particle_pos = to_screen_coords(particle_pos)
        r = (particle[7] / EarthMass) ** (1 / 3) * 10
        r = min(25, r)
        print(r)
        pygame.draw.circle(screen, (255, 0, 0), particle_pos, r)

    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

    # Move to the next timestep if not paused
    if not paused:
        current_index = (current_index + 1) % len(unique_timesteps)

pygame.quit()