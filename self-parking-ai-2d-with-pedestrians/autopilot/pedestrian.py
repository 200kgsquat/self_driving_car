import pygame as pg
from random import randint, uniform
from math import sin, cos, radians, atan2, degrees

__all__ = "Pedestrian"


class Pedestrian:
    """Pedestrian model that walks on the road area of the parking lot"""

    # Road zone boundaries (central area, avoiding edges where parked cars are)
    ROAD_LEFT = 140
    ROAD_RIGHT = 1180
    ROAD_TOP = 140
    ROAD_BOTTOM = 628

    def __init__(self, position=(0, 0), scale=1):
        sprite = pg.image.load(f"autopilot/sprites/pedestrian.png")
        rect = sprite.get_rect()
        w, h = round(rect.width * scale), round(rect.height * scale)
        self.sprite = pg.transform.scale(sprite, (w, h))
        self.width = w
        self.height = h
        self.scale = scale

        self.position = pg.math.Vector2(*position)
        self.velocity = pg.math.Vector2(0, 0)
        self.angle = uniform(0, 360)
        self.speed = uniform(20.0, 40.0) * scale
        self.max_speed = 40.0 * scale
        self.target_angle = self.angle  # for smooth steering

        self.is_alive = True
        self.hit = False

        # Movement timers (in seconds, using dt)
        self.move_timer = 0.0
        self.move_duration = uniform(1.0, 3.0)  # Move for 1-3 seconds
        self.wait_timer = 0.0
        self.wait_duration = uniform(0.5, 2.0)  # Wait 0.5-2 seconds

        # State: True = moving, False = waiting
        self.is_moving = True

        # Collision rectangle
        self.rect = pg.Rect(
            self.position.x - self.width // 2,
            self.position.y - self.height // 2,
            self.width,
            self.height
        )

    def _choose_new_direction(self):
        """Choose a new random walking direction"""
        self.target_angle = uniform(0, 360)
        self.move_duration = uniform(1.0, 3.0)
        self.move_timer = 0.0

    def _steer_away_from_edge(self):
        """Steer pedestrian back towards the center of the road"""
        cx = (self.ROAD_LEFT + self.ROAD_RIGHT) / 2
        cy = (self.ROAD_TOP + self.ROAD_BOTTOM) / 2
        dx = cx - self.position.x
        dy = cy - self.position.y
        self.target_angle = degrees(atan2(dy, dx))

    def _update(self, dt, bounds_rect):
        """Update pedestrian position and state using real-time dt"""
        if dt <= 0:
            return

        if self.is_moving:
            # Move phase
            self.move_timer += dt

            # Check if approaching road edge - steer back
            margin = 80
            if (self.position.x < self.ROAD_LEFT + margin or
                self.position.x > self.ROAD_RIGHT - margin or
                self.position.y < self.ROAD_TOP + margin or
                self.position.y > self.ROAD_BOTTOM - margin):
                self._steer_away_from_edge()

            # Smoothly rotate towards target angle (max 90 deg/s)
            angle_diff = (self.target_angle - self.angle + 540) % 360 - 180
            max_rotation = 90.0 * dt
            if abs(angle_diff) > max_rotation:
                angle_diff = max_rotation if angle_diff > 0 else -max_rotation
            self.angle += angle_diff

            # Calculate velocity based on angle
            rad_angle = radians(self.angle)
            self.velocity.x = self.speed * cos(rad_angle)
            self.velocity.y = self.speed * sin(rad_angle)

            # Calculate new position
            new_x = self.position.x + self.velocity.x * dt
            new_y = self.position.y + self.velocity.y * dt

            # Clamp to road zone
            half_w = self.width // 2
            half_h = self.height // 2
            clamped_x = max(self.ROAD_LEFT + half_w, min(new_x, self.ROAD_RIGHT - half_w))
            clamped_y = max(self.ROAD_TOP + half_h, min(new_y, self.ROAD_BOTTOM - half_h))

            # If clamped, steer back
            if clamped_x != new_x or clamped_y != new_y:
                self._steer_away_from_edge()

            self.position.x = clamped_x
            self.position.y = clamped_y

            # Update collision rectangle
            self.rect.x = int(self.position.x - self.width // 2)
            self.rect.y = int(self.position.y - self.height // 2)

            # Time to switch to waiting?
            if self.move_timer >= self.move_duration:
                self.is_moving = False
                self.wait_timer = 0.0
                self.wait_duration = uniform(0.5, 2.0)
                self.velocity.x = 0
                self.velocity.y = 0
        else:
            # Wait phase
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                self.is_moving = True
                self._choose_new_direction()
                self.speed = uniform(20.0, 40.0) * self.scale

    def _check_collision_with_car(self, car_collision_points):
        """Check if car's collision points overlap with pedestrian"""
        for point in car_collision_points:
            if self.rect.collidepoint(point):
                self.hit = True
                self.is_alive = False
                return True
        return False

    def move(self, dt, bounds_rect=None, car_collision_points=None):
        """Move pedestrian and check for collisions"""
        if self.is_alive:
            self._update(dt, bounds_rect)

            if car_collision_points is not None:
                self._check_collision_with_car(car_collision_points)

    def draw(self, screen):
        """Draw pedestrian on screen"""
        if self.is_alive:
            # Draw pedestrian sprite
            rotated_sprite = pg.transform.rotate(self.sprite, -self.angle)
            rect = rotated_sprite.get_rect()
            screen.blit(rotated_sprite, self.position - pg.math.Vector2(rect.width / 2, rect.height / 2))
        else:
            # Draw "hit" indicator
            pg.draw.circle(screen, (255, 0, 0), (int(self.position.x), int(self.position.y)), 10)