import pygame as pg
from random import randint, shuffle
from autopilot.pedestrian import Pedestrian

__all__ = ("SmallParking", "LargeParking")


class BaseParking:
    # -----------------------------------------------------------------
    # Pedestrian spawning configuration
    # -----------------------------------------------------------------
    # Total number of pedestrians to spawn in the simulation.
    PEDESTRIAN_COUNT = 4
    # Distance from a parking spot edge where pedestrians may appear.
    PEDESTRIAN_MARGIN = 30

    def _random_position_near_spot(self, rect, margin=None):
        """Return a random position outside *rect* but within *margin* pixels.

        ``rect`` is a tuple ``(x, y, w, h)`` describing a parking space.
        The function picks one of the four sides of the rectangle and places
        the pedestrian *margin* pixels away from that side, with a random offset
        along the side. The resulting coordinate is clamped to the map bounds
        (``MAP_WIDTH``/``MAP_HEIGHT``) to avoid out‑of‑range positions.
        """
        if margin is None:
            margin = self.PEDESTRIAN_MARGIN
        x, y, w, h = rect
        side = randint(0, 3)  # 0:left, 1:right, 2:top, 3:bottom
        if side == 0:  # left side, place to the left of the spot
            pos_x = x - margin
            pos_y = randint(y, y + h)
        elif side == 1:  # right side
            pos_x = x + w + margin
            pos_y = randint(y, y + h)
        elif side == 2:  # top side
            pos_y = y - margin
            pos_x = randint(x, x + w)
        else:  # bottom side
            pos_y = y + h + margin
            pos_x = randint(x, x + w)
        # Clamp to map bounds
        pos_x = max(0, min(pos_x, self.MAP_WIDTH))
        pos_y = max(0, min(pos_y, self.MAP_HEIGHT))
        return (pos_x, pos_y)

    def _spawn_pedestrians(self):
        """Spawn a fixed number of pedestrians near parking spots.

        Pedestrians are placed *outside* a randomly chosen parking space but
        within ``PEDESTRIAN_MARGIN`` pixels of its edge. This keeps them close to
        the spots without occupying the spot itself, satisfying the requirement
        that they do not interfere with car spawning.
        """
        self.pedestrians = []
        # Ensure we have at least one parking spot defined.
        if not hasattr(self, "spaces") or not self.spaces:
            return
        spots = list(self.spaces.values())
        for _ in range(self.PEDESTRIAN_COUNT):
            rect = spots[randint(0, len(spots) - 1)]
            pos = self._random_position_near_spot(rect)
            self.pedestrians.append(
                Pedestrian(
                    position=pos,
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

    # Map dimensions (must match Simulation window size)
    MAP_WIDTH = 1320
    MAP_HEIGHT = 768

    def __init__(self, pedestrian_count=5):
        # Load background image using an absolute path relative to this file to avoid
        # FileNotFoundError when the working directory is the project root.
        import os
        img_path = os.path.join(os.path.dirname(__file__), "sprites", "small-parking.png")
        self.background = pg.image.load(img_path)

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

        # Initialize a valid random start position and a normalized angle
        self.start_position = self._get_random_start_position()
        self.start_angle = self._get_random_start_angle()

        # Preserve backward compatibility but enforce the internal constant
        # for the actual number of pedestrians used in the simulation.
        self.pedestrian_count = self.PEDESTRIAN_COUNT

        self._init_parking()
        self.randomize()

        # -----------------------------------------------------------------
        # Configure a single pre‑determined empty parking spot to serve as the
        # default target. The user now wants only **one** reserved spot (instead
        # of the previous four side‑based empty spots plus an extra one).
        # We therefore replace the side‑based empty spots with a single index.
        # -----------------------------------------------------------------
        default_idx = 5  # The spot the user wants as the sole target.
        # Override the list of empty side indices with only the default.
        self.empty_side_idxs = [default_idx]
        # Re‑compute the occupied (parked) indices accordingly.
        self.parked_idxs = [i for i in self.spaces if i not in self.empty_side_idxs]
        # Set the target to this index and compute its centre position.
        self.target_idx = default_idx
        x, y, w, h = self.spaces[default_idx]
        self.target_position = (x + w / 2, y + h / 2)

    # -----------------------------------------------------------------
    # Helper methods to generate a valid random start position and angle.
    # -----------------------------------------------------------------
    def _get_random_start_position(self):
        """Return a random position within a small central zone.

        Previously the start position could be anywhere inside the map margins,
        which occasionally placed the car on a parking spot or too close to
        the edges. To improve training stability we now restrict the spawn to a
        configurable central square (default 200 × 200 pixels) around the map
        centre. This zone is still checked against pedestrians and parking
        spaces via ``is_position_safe_for_spawn``.
        """
        # Define the size of the central zone (half‑width/height from centre)
        zone_half = 100  # 200 px square total
        centre_x = self.MAP_WIDTH // 2
        centre_y = self.MAP_HEIGHT // 2
        min_x = centre_x - zone_half
        max_x = centre_x + zone_half
        min_y = centre_y - zone_half
        max_y = centre_y + zone_half
        # Generate a random point inside this central zone
        x = randint(min_x, max_x)
        y = randint(min_y, max_y)
        return (x, y)

    def _get_random_start_angle(self):
        """Return a normalized random angle in the range [0, 360)."""
        angle = randint(-180, 180)
        return angle % 360

    def _init_parking(self):
        coords = (
            [(123 + x, 659, 54, 98) for x in range(0, 1021, 60)] +
            [(1143 - x, 23, 54, 98) for x in range(0, 1021, 60)] +
            [(1199, 603 - y, 98, 54) for y in range(0, 481, 60)] +
            [(24, 123 + y, 98, 54) for y in range(0, 481, 60)]
        )

        self.spaces = {i: coords[i] for i in range(self.CAPACITY)}

        import os
        sprites_dir = os.path.join(os.path.dirname(__file__), "sprites")
        for i in range(self.CAPACITY):
            angle = 90 if i <= 17 else 180 if i <= 26 else 270 if i <= 44 else 0
            sprite_path = os.path.join(sprites_dir, f"car{i+1}.png")
            sprite = pg.image.load(sprite_path)
            self.cars_sprites.append(pg.transform.rotate(sprite, angle))

        # -----------------------------------------------------------------
        # Determine a single fixed parking spot for each side of the map.
        # -----------------------------------------------------------------
        # We iterate over all spot indices in order and pick the first spot
        # that belongs to each side (left, right, top, bottom). The side is
        # inferred from the rectangle geometry using the helper method
        # ``_side_of_idx`` defined later in the class.
        self.side_spot_idxs: dict = {}
        for idx in self.spaces:
            side = self._side_of_idx(idx)
            if side not in self.side_spot_idxs:
                self.side_spot_idxs[side] = idx
            if len(self.side_spot_idxs) == 4:
                break
        # Store the four fixed empty indices for quick access.
        self.empty_side_idxs = list(self.side_spot_idxs.values())

    # -----------------------------------------------------------------
    # Helper to determine if a position is blocked by map boundaries or a pedestrian.
    # -----------------------------------------------------------------
    def is_position_blocked(self, x, y, margin=0):
        """Return True if (x, y) is outside the map, collides with a pedestrian, or lies within a parking spot.

        margin: optional extra margin from the edges to consider unsafe.
        """
        if x < margin or x > self.MAP_WIDTH - margin or y < margin or y > self.MAP_HEIGHT - margin:
            return True
        # Check pedestrian proximity
        if self._is_position_occupied_by_pedestrian(x, y):
            return True
        # Check if the point falls inside any parking space rectangle (with optional margin)
        for rect in self.spaces.values():
            rx, ry, rw, rh = rect
            if (rx - margin) <= x <= (rx + rw + margin) and (ry - margin) <= y <= (ry + rh + margin):
                return True
        return False

    def _is_position_occupied_by_pedestrian(self, x, y, radius=30):
        """Return True if the given (x, y) is within ``radius`` of any pedestrian.

        This helper is used by ``is_position_blocked`` to avoid spawning the car on
        top of a pedestrian. It iterates over ``self.pedestrians`` and checks the
        Euclidean distance between the point and each pedestrian's position.
        """
        for ped in getattr(self, "pedestrians", []):
            # Pedestrian has a ``position`` attribute (x, y)
            px, py = ped.position
            if (x - px) ** 2 + (y - py) ** 2 <= radius ** 2:
                return True
        return False

    def _side_of_idx(self, idx):
        """Return the side of the map ('left', 'right', 'top', 'bottom') for a given parking spot index.

        The parking layout is built in ``_init_parking`` where the order of spots is:
        1. Bottom side (y ≈ 659)
        2. Top side (y ≈ 23)
        3. Right side (x ≈ 1199)
        4. Left side (x ≈ 24)

        We infer the side from the rectangle geometry: horizontal spots (width > height)
        are on the left/right edges, vertical spots (height > width) are on the top/bottom.
        The exact edge is distinguished by the x or y coordinate.
        """
        rect = self.spaces[idx]
        x, y, w, h = rect
        if w > h:  # left or right side (wide rectangle)
            # Left side rectangles are positioned near the left map border (small x)
            return "left" if x < self.MAP_WIDTH // 2 else "right"
        else:  # top or bottom side (tall rectangle)
            return "top" if y < self.MAP_HEIGHT // 2 else "bottom"

    def randomize(self):
        """Randomize the parking layout while keeping the four fixed empty spots.

        The fixed empty spots are determined once in ``_init_parking`` and stored in
        ``self.empty_side_idxs``. Here we simply shuffle the full list of indices and
        mark every index *except* those fixed ones as occupied (parked). This guarantees
        that the same four spots – one per side – are always available for target
        selection throughout the entire simulation.
        """
        idxs = list(self.spaces.keys())
        shuffle(idxs)

        # Ensure we have the fixed empty indices; if not (unlikely), fall back to the
        # previous per‑side selection logic to avoid crashes.
        # If the list of empty indices is not defined or empty, fall back to the
        # original behaviour of selecting one empty spot per side. This allows
        # callers (like __init__) to pre‑define a custom set of empty spots –
        # for example a single reserved spot – without it being overwritten.
        if not hasattr(self, "empty_side_idxs") or not self.empty_side_idxs:
            empty_per_side = {}
            for idx in idxs:
                side = self._side_of_idx(idx)
                if side not in empty_per_side:
                    empty_per_side[side] = idx
                if len(empty_per_side) == 4:
                    break
            self.empty_side_idxs = list(empty_per_side.values())

        # All indices except the fixed empty ones are considered occupied.
        self.parked_idxs = [i for i in idxs if i not in self.empty_side_idxs]
        self.target_idx = None
        self.target_position = None

        self._spawn_pedestrians()
        # Refresh start position and angle for each new layout
        self.start_position = self._get_random_start_position()
        self.start_angle = self._get_random_start_angle()

    def is_position_safe_for_spawn(self, margin=30, attempts=100):
        """Return a spawn position that is not blocked by walls or pedestrians.

        Tries up to ``attempts`` random positions using ``_get_random_start_position``.
        If all attempts are blocked, falls back to the centre of the map.
        """
        for _ in range(attempts):
            pos = self._get_random_start_position()
            if not self.is_position_blocked(*pos, margin=margin):
                return pos
        # Fallback to centre if no safe spot found
        return (self.MAP_WIDTH // 2, self.MAP_HEIGHT // 2)

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
        """Select a random side and use its pre‑selected empty spot as the target.

        ``self.empty_side_idxs`` holds exactly four indices – one per side – that were left empty
        during ``randomize``. We randomly choose one of these sides, set it as the target, and keep the
        remaining three empty spots untouched (they remain available for future episodes).
        """
        if not hasattr(self, "empty_side_idxs") or not self.empty_side_idxs:
            return False
        import random
        idx = random.choice(self.empty_side_idxs)
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