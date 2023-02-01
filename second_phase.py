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

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        sprite_path = 'maps/red_car.png' if i == 1 else 'maps/blue_car.png'
        cars.append(Car(sprite_path))

    clock = pygame.time.Clock()
    MAP, PICKLE_FOLDER = get_constants_main(MAP_INDEX)
    game_map = pygame.image.load(MAP).convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if (car.speed - 2 >= 12):
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1

                car.update(game_map, 2, cars[1 - i], screen, with_detection)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw_only_car(screen)

        # Display Info
        # pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, 122, 80))

        # text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        # text_rect = text.get_rect()
        # text_rect.center = (60, 35)
        # screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

        # print("red: ")
        # print(cars[0].distance)
        # print(cars[0].time, end="\n\n")
        # print("blue: ")
        # print(cars[1].distance)
        # print(cars[1].time, end="\n\n")
