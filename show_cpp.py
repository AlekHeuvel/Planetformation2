import pygame
import pandas as pd
import numpy as np

# Pygame setup
pygame.init()
width, height = 1400, 750
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()
AU = 149.6e9

# Function to convert simulation coordinates to screen coordinates
def to_screen_coords(pos, scale=20*AU / height):
    coords = int(width / 2 + pos[0] / scale), int(height / 2 - pos[1] / scale)
    return coords

def read_particle_data(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, header=None)
    df.columns = ['Timestep', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ', 'Mass']

    return df


data = read_particle_data("data.csv")
running = True
for i in range(data['Timestep'].max() + 1):

    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the particles
    for particle in data[data['Timestep'] == i].values:
        particle_pos = particle[1:4]
        particle_pos = to_screen_coords(particle_pos)
        #print(f"{particle_pos}")
        pygame.draw.circle(screen, (255, 0, 0), particle_pos, 5)

    pygame.display.flip()
    # Draw the sun at the center
    # pygame.draw.circle(screen, (255, 255, 0), to_screen_coords(np.array([0, 0])), 10)

    # Update the display


    # Cap the frame rate
    clock.tick(10)
pygame.quit()