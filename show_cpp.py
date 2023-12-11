import pygame
import pandas as pd

# Pygame setup
pygame.init()
width, height = 1400, 750
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()
AU = 149.6e9

# Function to convert simulation coordinates to screen coordinates
def to_screen_coords(pos, scale=4*AU / height):
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
for timestep in unique_timesteps:

    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the particles
    for particle in data[data['Timestep'] == timestep].values:
        particle_pos = particle[1:3]  # Only X and Y coordinates
        particle_pos = to_screen_coords(particle_pos)
        pygame.draw.circle(screen, (255, 0, 0), particle_pos, 5)

    # Draw the sun at the center (optional)
    # pygame.draw.circle(screen, (255, 255, 0), to_screen_coords(np.array([0, 0])), 10)

    pygame.display.flip()

    # Cap the frame rate
    clock.tick(100)

pygame.quit()