import pickle
import random
import sys
import neat
import pygame
from car import Car
from constants import MAP_INDEX, WIDTH, HEIGHT
from second_phase import run_simulation_phase_two
from util import *

nets = []


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
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

        cars.append(Car('maps/red_car.png'))

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
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

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map, 1)
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
                car.draw(screen)

        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, 180, 100))
        text_rect = text.get_rect()
        text_rect.center = (100, 25)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, 50)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


def detect_road(image_path):
    img = load_image(image_path)
    gray_img = image_gray(img)
    th, bin_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)
    select_roi(img, bin_img)


def apply_mask(img, mask):
    orignal_gray_frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = mask / 255

    mask_array = np.asarray(mask, dtype=np.float64)
    original_array = np.asarray(orignal_gray_frame, dtype=np.float64)

    subtraction_between_frames = np.multiply(mask_array, original_array)
    display_image(subtraction_between_frames)


if __name__ == "__main__":
    # Load Config
    config_path = "config.txt"
    MAP, PICKLE_FOLDER = get_constants_main(MAP_INDEX)

    with open(PICKLE_FOLDER + "red.pkl", "rb") as f:
        blue_genome = pickle.load(f)

    with open(PICKLE_FOLDER + "blue.pkl", "rb") as f:
        red_genome = pickle.load(f)

    genomes = [(1, red_genome), (2, blue_genome)]

    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    detect_road(MAP)

    # population = neat.Population(config)
    # population.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # population.add_reporter(stats)
    # generations = 40

    # winner = population.run(run_simulation, generations)
    run_simulation_phase_two(genomes, config, False)

    # with open(PICKLE_FOLDER + "red.pkl", "wb") as f:
    #     pickle.dump(winner, f)