import pygame

from constants import *
from util import *


class Car:

    def __init__(self, car_image_path):
        self.car_image_path = car_image_path
        self.car = pygame.image.load(car_image_path).convert_alpha()
        self.car = pygame.transform.scale(self.car, (CAR_SIZE_X, CAR_SIZE_Y))
        self.fire = pygame.image.load('maps/fire.png').convert_alpha()
        self.fire = pygame.transform.scale(self.fire, (FIRE_SIZE_X, FIRE_SIZE_Y))

        self.rotated_car = self.car

        BLUE_POSITION, RED_POSITION = get_constants_car(MAP_INDEX)
        self.position = BLUE_POSITION if car_image_path == 'maps/blue_car.png' else RED_POSITION

        self.angle = 0
        self.speed = 0

        self.speed_set = False

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []  # List of Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

        self.detected_map = pygame.image.load('maps/detected_map.png').convert()
        self.frame_iterator = 0

    def draw(self, screen):
        screen.blit(self.rotated_car, self.position)
        self.draw_radar(screen)

    def draw_only_car(self, screen):
        screen.blit(self.rotated_car, self.position)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self):
        self.alive = True
        for point in self.corners:
            if self.detected_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not self.detected_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

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
                MAP, PICKLE_FOLDER = get_constants_main(MAP_INDEX)
                img = load_image(MAP)
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), 2)
                cv2.circle(img, (int(left_top[0]), int(left_top[1])), 5, (255, 0, 255), 2)
                cv2.circle(img, (int(right_top[0]), int(right_top[1])), 5, (255, 0, 255), 2)
                cv2.circle(img, (int(left_bottom[0]), int(left_bottom[1])), 5, (255, 0, 255), 2)
                cv2.circle(img, (int(right_bottom[0]), int(right_bottom[1])), 5, (255, 0, 255), 2)
                display_image(img)
                screen.blit(self.fire, (self.center[0] - 25, self.center[1] - 30))
                pygame.display.update()
                pygame.time.wait(700)
                self.alive = False

    def check_collision_with_another_car_no_detection(self, another_car, screen):
        if not another_car.alive:
            return
        for point in self.corners:
            if self.does_points_lie_on_rectangle(point, another_car):
                screen.blit(self.fire, (self.center[0] - 25, self.center[1] - 30))
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

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, phase_index, another_car=None, screen=None, with_detection=False):
        # skip_frames = 5 if with_detection else 1

        coords = None
        if phase_index != 1 and self.alive and with_detection and self.frame_iterator % 10 == 0:
            string_image = pygame.image.tostring(screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, (WIDTH, HEIGHT), 'RGB')
            tmp_arr = pygame.surfarray.array3d(temp_surf)

            image = cv2.rotate(tmp_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.flip(image, 0)

            coords = detect_another_car(image, get_corners(self.center, self.angle, 0.5 * CAR_SIZE_X))

        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_car = self.rotate_center(self.car, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        self.corners = get_corners(self.center, self.angle, length)

        if self.frame_iterator % 10 == 0:
            self.check_collision()
            if not with_detection and phase_index != 1 and self.alive:
                self.check_collision_with_another_car_no_detection(another_car, screen)
            elif phase_index != 1 and self.alive and coords is not None:
                x, y, w, h = coords
                self.check_collision_with_another_car([x + w // 2, y + h // 2], screen, another_car.angle)

        self.frame_iterator += 1

        self.radars.clear()

        for d in range(-90, 120, 45):
            if phase_index == 1:
                self.check_radar(d)
            else:
                self.check_radar(d)
                # self.check_radar_for_another_car(d, another_car)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
