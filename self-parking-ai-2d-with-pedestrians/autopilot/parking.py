import pygame as pg
from random import randint, shuffle
from autopilot.pedestrian import Pedestrian

__all__ = ("SmallParking", "LargeParking")


class BaseParking:

    def _spawn_pedestrians(self):
        self.pedestrians = []

        for _ in range(self.pedestrian_count):
            self.pedestrians.append(
                Pedestrian(
                    position=(randint(200, 1120), randint(200, 568)),
                    scale=0.8
                )
            )

    def update_pedestrians(self, dt, car_collision_points=None):
        bounds = pg.Rect(0, 0, 1320, 768)

        for p in self.pedestrians:
            p.move(dt, bounds, car_collision_points)

    def draw_pedestrians(self, screen):
        for p in self.pedestrians:
            p.draw(screen)


class SmallParking(BaseParking):
    CAPACITY = 54

    def __init__(self, pedestrian_count=5):
        self.background = pg.image.load("autopilot/sprites/small-parking.png")

        self.grass_color = (63, 155, 11, 255)
        self.markup_color = (255, 255, 255, 255)
        self.road_color = (80, 80, 80, 255)

        self.pointers_color = (242, 188, 10, 255)
        self.hover_color = (0, 255, 0, 255)
        self.road_pointers_color = (161, 134, 45, 255)

        self.spaces = {}
        self.cars_sprites = []

        self.parked_idxs = []
        self.target_idx = None
        self.target_position = None

        self.start_angle = 0
        self.start_position = (660, 384)

        self.pedestrian_count = pedestrian_count

        self._init_parking()
        self.randomize()

    def _init_parking(self):
        coords = (
            [(123 + x, 659, 54, 98) for x in range(0, 1021, 60)] +
            [(1143 - x, 23, 54, 98) for x in range(0, 1021, 60)] +
            [(1199, 603 - y, 98, 54) for y in range(0, 481, 60)] +
            [(24, 123 + y, 98, 54) for y in range(0, 481, 60)]
        )

        self.spaces = {i: coords[i] for i in range(self.CAPACITY)}

        for i in range(self.CAPACITY):
            angle = 90 if i <= 17 else 180 if i <= 26 else 270 if i <= 44 else 0
            sprite = pg.image.load(f"autopilot/sprites/car{i+1}.png")
            self.cars_sprites.append(pg.transform.rotate(sprite, angle))

    def randomize(self):
        idxs = list(self.spaces.keys())
        shuffle(idxs)

        # Leave one random spot empty so the car has a target
        self.parked_idxs = idxs[:-1]
        self.target_idx = None
        self.target_position = None

        self._spawn_pedestrians()

    def select_parking_space(self, mouse_pos):
        for idx, rect in self.spaces.items():
            if pg.Rect(rect).collidepoint(mouse_pos):

                if self.target_idx is not None and self.target_idx not in self.parked_idxs:
                    self.parked_idxs.append(self.target_idx)

                if idx in self.parked_idxs:
                    self.parked_idxs.remove(idx)

                self.target_idx = idx
                x, y, w, h = rect
                self.target_position = (x + w / 2, y + h / 2)
                break

    def pick_random_target(self):
        """Select a random unoccupied parking spot as the target"""
        available = [idx for idx in self.spaces.keys() if idx not in self.parked_idxs]
        if not available:
            return False
        import random
        idx = random.choice(available)
        self.target_idx = idx
        x, y, w, h = self.spaces[idx]
        self.target_position = (x + w / 2, y + h / 2)
        return True

    def draw_hover(self, screen, mouse_pos):
        for rect in self.spaces.values():
            if pg.Rect(rect).collidepoint(mouse_pos):
                pg.draw.rect(screen, self.hover_color, rect, 3)
                break

    @staticmethod
    def get_center(rect, car):
        x, y, w, h = rect
        cw, ch = car.get_size()
        return (x + w / 2 - cw / 2, y + h / 2 - ch / 2)

    def draw(self, screen):
        screen.blit(self.background, (0, 0))

        if self.target_idx is not None:
            pg.draw.rect(screen, self.pointers_color, self.spaces[self.target_idx], 5)

        for i in self.parked_idxs:
            rect = self.spaces[i]
            car = self.cars_sprites[i]
            screen.blit(car, self.get_center(rect, car))