import pygame

from constants import CAR_SIZE_X, CAR_SIZE_Y, FIRE_SIZE_X, MAP_INDEX, FIRE_SIZE_Y, BORDER_COLOR, WIDTH, HEIGHT
from util import *


class Car:

    def __init__(self, car_image_path):
        # Load Car Sprite and Rotate
        self.car = pygame.image.load(car_image_path).convert_alpha()  # Convert Speeds Up A Lot
        self.car = pygame.transform.scale(self.car, (CAR_SIZE_X, CAR_SIZE_Y))
        self.fire = pygame.image.load('maps/fire.png').convert_alpha()  # Convert Speeds Up A Lot
        self.fire = pygame.transform.scale(self.fire, (FIRE_SIZE_X, FIRE_SIZE_Y))

        self.rotated_car = self.car

        BLUE_POSITION, RED_POSITION = get_constants_car(MAP_INDEX)
        self.position = BLUE_POSITION if car_image_path == 'maps/blue_car.png' else RED_POSITION

        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

        self.detected_map = pygame.image.load('maps/detected_map.png').convert()

    def draw(self, screen):
        screen.blit(self.rotated_car, self.position)
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_only_car(self, screen):
        screen.blit(self.rotated_car, self.position)

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if self.detected_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not self.detected_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def check_collision_with_another_car(self, center_of_another_car, screen, angle):
        left_top, right_top, left_bottom, right_bottom = get_corners(center_of_another_car, angle,
                                                                     0.5 * CAR_SIZE_X)

        for point in self.corners:
            if check_if_center_is_in_corners(left_top[0], left_top[1],
                                             right_top[0], right_top[1],
                                             left_bottom[0], left_bottom[1],
                                             right_bottom[0], right_bottom[1], point[0], point[1]):

                screen.blit(self.fire, point)
                pygame.display.update()
                pygame.time.wait(300)
                self.alive = False

    def check_collision_with_another_car_no_detection(self, another_car, screen):
        if not another_car.alive:
            return
        for point in self.corners:
            if self.does_points_lie_on_rectangle(point, another_car):
                screen.blit(self.fire, point)
                pygame.display.update()
                pygame.time.wait(300)
                self.alive = False
                break


    def does_points_lie_on_rectangle(self, point, another_car):
        left_top, right_top, left_bottom, right_bottom = get_corners(another_car.center, another_car.angle,
                                                                     0.5 * CAR_SIZE_X)

        return check_if_center_is_in_corners(left_top[0], left_top[1],
                     right_top[0], right_top[1],
                     left_bottom[0], left_bottom[1],
                     right_bottom[0], right_bottom[1],
                     point[0], point[1])

    def check_radar_for_another_car(self, degree, another_car):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not self.detected_map.get_at((x, y)) == BORDER_COLOR \
                and not self.does_points_lie_on_rectangle((x, y), another_car) \
                and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map, phase_index, another_car=None, screen=None, with_detection=False):

        coords = None
        if phase_index != 1 and self.alive and with_detection:
            string_image = pygame.image.tostring(screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, (WIDTH, HEIGHT), 'RGB')
            tmp_arr = pygame.surfarray.array3d(temp_surf)

            image = cv2.rotate(tmp_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.flip(image, 0)

            coords = detect_another_car(image, get_corners(self.center, self.angle, 0.5 * CAR_SIZE_X))

        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_car = self.rotate_center(self.car, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        self.corners = get_corners(self.center, self.angle, length)

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        if not with_detection and phase_index != 1 and self.alive:
            self.check_collision_with_another_car_no_detection(another_car, screen)
        elif phase_index != 1 and self.alive and coords is not None:
            x, y, w, h = coords
            self.check_collision_with_another_car([x + w // 2, y + h // 2], screen, another_car.angle)

        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            if phase_index == 1:
                self.check_radar(d, game_map)
            else:
                self.check_radar(d, game_map)
                # self.check_radar_for_another_car(d, another_car)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
