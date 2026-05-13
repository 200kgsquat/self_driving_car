import pygame as pg
import numpy as np
from pygame.math import Vector2
from math import sqrt, sin, cos, atan2, radians, degrees, copysign

__all__ = ("Car",)


class Car:
    """Kinematic car model with radars and reward system"""

    def __init__(
        self,
        spawn_position=(0.0, 0.0),
        spawn_angle=0,
        scale=1,
        show_collision=False,
        show_radars=False,
        show_score=False
    ):
        # Load car sprite using absolute path relative to this file
        import os
        sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "car0.png")
        sprite = pg.image.load(sprite_path)
        rect = sprite.get_rect()

        w, h = round(rect.width * scale), round(rect.height * scale)

        self.car_sprite = pg.transform.scale(sprite, (w, h))
        self.car_sprite_width = 0.5 * w - 5
        self.car_sprite_height = 0.5 * h - 10
        self.chassis_length = 0.03 * h

        # physics
        self.angle = spawn_angle
        self.position = Vector2(*spawn_position)
        self.velocity = Vector2(0.0, 0.0)
        self.acceleration = 0.0
        self.steering = 0.0

        # smoothing for NEAT outputs (prevents jerky movement)
        # Higher = more responsive, lower = smoother
        self.smoothed_direction = 0.0
        self.smoothed_rotation = 0.0
        # Reduce smoothing to keep responsiveness while still avoiding jerky jumps
        self.smooth_factor = 0.5
        # Increase limits to give the car more room to accelerate and steer
        self.max_velocity = 80.0 * scale
        self.max_acceleration = 5.0 * scale
        self.max_steering = 1.5 * scale
        self.max_radar_len = int(300 * scale)

        self.brake_deceleration = 10.0 * scale
        self.free_deceleration = 2.0 * scale

        # state
        self.is_alive = True
        # store debug flags for later resets
        self._show_collision = show_collision
        self._show_radars = show_radars
        self._show_score = show_score
        # Counter for consecutive stationary steps (used for progressive penalty)
        self.stationary_counter = 0
        # Flag to ensure parking bonus is applied only once per episode
        self.parked_bonus_given = False

    def reset(self, position, angle):
        """Reset the car to a new start position and angle.
        This is used during training to re‑initialize a car for a new genome.
        """
        self.position = pg.Vector2(position)
        self.angle = angle % 360
        self.velocity = pg.Vector2(0, 0)
        self.acceleration = 0.0
        self.steering = 0.0
        self.score = 0.0
        self.movement_score = 0.0
        self.distance_score = 0.0
        self.is_alive = True
        # Reset any other dynamic state if needed
        self.parked = False
        # Reset parking bonus flag for a new episode
        self.parked_bonus_given = False

        # sensors
        self.radars_data = np.zeros(8, np.float32)
        self.navigation = np.zeros(5, np.float32)
        self.radars = []
        self.collision_points = np.zeros((4, 2), dtype=np.int32)

        # reward
        self.target_distance = 0.0
        self.start_distance = 0.0
        self.distance_score = 0.0
        self.movement_score = 0.0
        self.score = 0.0

        # debug flags restored from stored defaults
        self.show_collision_points = self._show_collision
        self.show_radars = self._show_radars
        self.show_score = self._show_score

    # ---------------- PHYSICS ----------------

    def _update(self, movement, dt):

        # Smooth NEAT outputs to prevent jerky movement
        raw_dir = movement["direction"]
        raw_rot = movement["rotation"]

        if isinstance(raw_dir, str):
            raw_dir = {"forward": 1, "backward": -1, "neutral": 0}.get(raw_dir, 0)
        if isinstance(raw_rot, str):
            raw_rot = {"right": 1, "left": -1, "neutral": 0}.get(raw_rot, 0)

        self.smoothed_direction += (raw_dir - self.smoothed_direction) * self.smooth_factor
        self.smoothed_rotation += (raw_rot - self.smoothed_rotation) * self.smooth_factor

        dir_val = self.smoothed_direction
        rot_val = self.smoothed_rotation

        if dir_val > 0.3:
            self.acceleration += dt if self.velocity.x >= 0 else self.brake_deceleration

        elif dir_val < -0.3:
            self.acceleration -= dt if self.velocity.x <= 0 else self.brake_deceleration

        else:
            if abs(self.velocity.x) > dt * self.free_deceleration:
                self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
            else:
                self.acceleration = -self.velocity.x / max(dt, 1e-6)

        # steering
        if rot_val > 0.3:
            self.steering -= self.max_steering * dt
        elif rot_val < -0.3:
            self.steering += self.max_steering * dt
        else:
            self.steering = 0

        self.velocity.x += self.acceleration * dt
        self._clamp()

        if abs(self.steering) > 1e-6:
            r = self.chassis_length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / r
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

    def _clamp(self):
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))

    def _stop(self):
        self.is_alive = False
        self.velocity.x = 0
        self.acceleration = 0
        self.steering = 0

    # ---------------- COLLISION ----------------

    def _compute_collision_points(self):
        sin_a = sin(radians(-self.angle))
        cos_a = cos(radians(-self.angle))

        left = self.position.x - self.car_sprite_width
        right = self.position.x + self.car_sprite_width
        top = self.position.y + self.car_sprite_height
        bottom = self.position.y - self.car_sprite_height

        pts = [(right, top), (right, bottom), (left, top), (left, bottom)]
        res = []

        for x, y in pts:
            ox, oy = x - self.position.x, y - self.position.y
            nx = self.position.x + ox * cos_a - oy * sin_a
            ny = self.position.y + ox * sin_a + oy * cos_a
            res.append((int(nx), int(ny)))

        self.collision_points = np.array(res, dtype=np.int32)

    def _check_collision(self, screen, surface, pedestrians=None):
        """Detect collisions with map boundaries and pedestrians.

        The original implementation stopped the car on any wall collision and
        also stopped the car when hitting a pedestrian. This caused stationary
        cars (velocity ≈ 0) to be marked as dead even when they were merely
        touching a wall or a pedestrian, which interfered with learning.

        Updated logic:
        * Wall collisions only terminate the car if it is moving (speed > 0.1).
          Otherwise the car is allowed to remain alive – the situation is
          treated as a harmless contact.
        * Pedestrian collisions now only apply a penalty to ``movement_score``
          without stopping the car, encouraging the network to avoid them but
          not penalising the entire episode.
        """
        # Check collision with walls/obstacles
        for p in self.collision_points:
            if not self._safe_position(p, screen, surface):
                self.movement_score -= 10
                # Stop only if the car is actually moving
                if abs(self.velocity.x) > 0.1:
                    self._stop()
                return

        # Pedestrian collisions – apply penalty but do not stop the car
        if pedestrians:
            for ped in pedestrians:
                if ped.is_alive:
                    for p in self.collision_points:
                        if ped.rect.collidepoint(p):
                            self.movement_score -= 5
                            # Do NOT stop the car here; allow continued learning
                            return

    # ---------------- RADARS ----------------

    def _compute_radars(self, screen, surface, pedestrians=None):
        """Compute radar distances safely.

        Any unexpected error (e.g., out‑of‑bounds access) is caught so that a
        single faulty genome does not crash the whole simulation. In case of
        an exception the radar is set to the maximum range.
        """
        angles = [radians(90 - self.angle - 45 * i) for i in range(8)]

        self.radars = []
        self.radars_data = np.zeros(8, np.float32)

        for i, ang in enumerate(angles):
            dist = self.max_radar_len
            x, y = 0, 0
            try:
                for d in range(1, self.max_radar_len):
                    x = int(self.position.x + d * cos(ang))
                    y = int(self.position.y + d * sin(ang))
                    if not self._safe_position((x, y), screen, surface):
                        dist = d
                        break
            except Exception as e:
                # Log the error for diagnostics and fall back to max distance
                print(f"[RADAR ERROR] {e} – using max range for radar {i}")
                dist = self.max_radar_len
            self.radars.append((x, y))
            self.radars_data[i] = dist / self.max_radar_len

    # ---------------- SAFE ----------------

    def _safe_position(self, position, screen, surface, limit=60):
        try:
            color = screen.get_at(position)
            return any([
                self._color_dist(color, surface.road_color) < limit,
                self._color_dist(color, surface.pointers_color) < limit,
                self._color_dist(color, surface.road_pointers_color) < limit
            ])
        except:
            return False

    @staticmethod
    def _color_dist(c1, c2):
        return sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # ---------------- TARGET ----------------

    def _compute_target_distance(self, surface):
        if not hasattr(surface, "target_position") or surface.target_position is None:
            self.target_distance = 0
            return

        dist = sqrt(
            (self.position.x - surface.target_position[0]) ** 2 +
            (self.position.y - surface.target_position[1]) ** 2
        )

        if self.start_distance == 0:
            self.start_distance = dist

        self.target_distance = max(0, 1 - dist / (self.start_distance + 1e-6))

    def update_movement_penalty(self):
        """Adjust movement score without penalising stationary behaviour.

        The original implementation applied a small base penalty (‑0.3) and a
        progressive penalty that grew with the number of consecutive frames the
        car remained effectively still. This discouraged the "stand‑still"
        strategy but also introduced a bias that could dominate the overall
        reward signal, making it harder for the NEAT algorithm to discover
        useful behaviours.

        The updated logic **removes** any negative reward for being stationary.
        Instead, we only grant a positive bonus when the car is moving and keep
        the ``stationary_counter`` for potential future use (e.g., analytics) but
        do not affect the score.
        """
        speed = abs(self.velocity.x) + abs(self.velocity.y)
        if speed < 0.5:
            # Car is effectively stationary – no penalty applied.
            self.stationary_counter += 1
            # Optionally, a very small neutral reward could be added here if
            # desired, but we keep the score unchanged to fully remove the
            # standing penalty.
        else:
            # Reset counter when moving and give a movement bonus.
            self.stationary_counter = 0
            self.movement_score += 0.8

    def _compute_score(self):
        """Calculate the car's score for the current tick.

        Adjusted to discourage the "stand‑still" strategy and provide stronger
        incentives for moving towards the target.
        """
        # 1. Apply stationary penalty / movement reward
        self.update_movement_penalty()

        # 3. Reward for forward motion (encourage higher speed)
        # forward_reward = 0.5 * max(0, self.velocity.x)
        # self.movement_score += forward_reward

        # 4. Small reward for active steering (encourage exploration)
        if abs(self.steering) > 0.1:
            self.movement_score += 0.05

        # 5. Distance reward – stronger scaling
        self.distance_score = 500 * self.target_distance

        # Bonus when the car reaches the target.
        # Use a stricter condition based on target progress and ensure the bonus
        # is granted only once per episode.
        if not self.parked_bonus_given and self.target_distance > 0.95:
            self.distance_score = 1000
            self.movement_score += 300   # larger bonus for successful parking
            self.parked_bonus_given = True
            self._stop()
            self.parked = True

        # Compute final score
        self.score = self.distance_score + self.movement_score

        # Guard against NaN / Inf – replace with a large negative penalty
        if np.isnan(self.score) or np.isinf(self.score):
            self.score = -100.0

    def _navigate_target(self, surface):
        if not hasattr(surface, "target_position") or surface.target_position is None:
            self.navigation = np.zeros(5, np.float32)
            return

        tx, ty = surface.target_position
        dx = tx - self.position.x
        dy = ty - self.position.y

        angle = degrees(atan2(dy, dx))

        forward = 1.0 if abs(angle - self.angle) < 90 else 0.0
        self.navigation = np.array([forward, 1 - forward, 0, 0, self.target_distance])

    # ---------------- MAIN ----------------

    def move(self, movement, dt, screen, surface, pedestrians=None):
        if not self.is_alive:
            return

        self._update(movement, dt)
        self._compute_collision_points()
        self._check_collision(screen, surface, pedestrians)
        self._compute_radars(screen, surface, pedestrians)
        self._compute_target_distance(surface)
        self._navigate_target(surface)
        self._compute_score()

    # ---------------- DRAW ----------------

    def draw(self, screen):
        rotated = pg.transform.rotate(self.car_sprite, self.angle)
        rect = rotated.get_rect()
        screen.blit(rotated, self.position - Vector2(rect.width / 2, rect.height / 2))