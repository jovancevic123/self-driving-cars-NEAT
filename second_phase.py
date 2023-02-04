import sys

import neat
import pygame

from car import Car
from constants import *
from util import get_constants_main


def run_simulation_phase_two(genomes, config, with_detection):
    print("SECOND PHASE STARTED !!!")
    global nets
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        sprite_path = 'maps/red_car.png' if i == 1 else 'maps/blue_car.png'
        cars.append(Car(sprite_path))

    clock = pygame.time.Clock()
    MAP, PICKLE_FOLDER = get_constants_main(MAP_INDEX)
    game_map = pygame.image.load(MAP).convert()

    global current_generation
    current_generation += 1

    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1

                car.update(2, cars[1 - i], screen, with_detection)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw_only_car(screen)

        pygame.display.flip()
        clock.tick(60)
