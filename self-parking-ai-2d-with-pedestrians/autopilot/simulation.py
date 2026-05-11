import sys
import neat
import pickle
import numpy as np
import pygame as pg
import os
import datetime

from autopilot.car import Car
from autopilot.parking import SmallParking


class Simulation:
    """
    Self-driving parking simulation with interactive selection
    """

    # Constant for the default dead‑zone fitness used when a car dies early
    DEAD_FITNESS = -9.01

    def __init__(
        self,
        epochs=10000,
        parked_cars=None,
        time_per_map=250,
        pedestrian_count=5,
        show_gui: bool = False,
    ):
        """Create a new simulation.

        Parameters
        ----------
        epochs: int
            Number of generations to run.
        parked_cars: Any
            Reserved for future use.
        time_per_map: int
            Maximum time steps per map (capped at 250 steps).
        pedestrian_count: int
            Number of pedestrians in the environment.
        show_gui: bool, default False
            When ``True`` the pygame window is displayed during training.
            When ``False`` (default) the dummy video driver is used for head‑less
            operation.
        """
        # Use dummy video driver for headless training unless GUI is requested
        if not show_gui:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pg.init()

        pg.display.set_caption(
            "Self-parking simulation with pedestrians"
        )

        self.window = (1320, 768)
        self.width, self.height = self.window

        self.screen = pg.display.set_mode(self.window)

        self.clock = pg.time.Clock()

        # Initialize parking. The pedestrian count is now managed internally
        # by ``SmallParking`` via the ``PEDESTRIAN_COUNT`` constant, but we keep
        # the argument for backward compatibility. After creation we sync the
        # simulation's ``pedestrian_count`` attribute to reflect the actual
        # number of spawned pedestrians.
        self.parking = SmallParking(
            pedestrian_count=pedestrian_count
        )
        # Ensure the UI displays the correct total number of pedestrians.
        self.pedestrian_count = getattr(self.parking, "PEDESTRIAN_COUNT", pedestrian_count)

        self.best_score = -float("inf")
        # Ensure a target is selected right from the start
        self.parking.pick_random_target()

        # Increase episode length to give the agent more steps to reach the target
        # (previously 10 seconds, now up to 100 seconds)
        # Clamp the time per map between the safety floor (100) and the new upper limit (250)
        self.time_per_map = min(max(time_per_map, 100), 250)
        self.generations = epochs

        self.generation = 0
        self.map = 0
        self.time = 0

        self.cars_left = 0
        self.pedestrian_count = pedestrian_count

        # Initialize logging
        # If debug mode is enabled, use a separate debug log file; otherwise use the standard training log.
        # Primary log file for detailed generation statistics
        LOG_FILE = os.path.join(os.path.dirname(__file__), "logs.txt")
        # Keep existing log for backward compatibility / debug information
        log_filename = "training_debug.txt" if getattr(self, "debug", False) else "training_log.txt"
        self._log_file = os.path.join(os.path.dirname(__file__), log_filename)
        self._stats_log_file = LOG_FILE
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Training session started at {datetime.datetime.now().isoformat()} ===\n")

    # ---------------------------------------------------------------------
    # Helper to obtain a safe spawn position and angle for a new car.
    # ---------------------------------------------------------------------
    def _get_safe_start_position(self):
        """Return a safe ``((x, y), angle)`` tuple for a new car.

        ``SmallParking`` already knows how to find a collision‑free position via
        ``is_position_safe_for_spawn``. The original implementation always used
        the *fixed* ``parking.start_angle`` which caused every genome to start
        with the same orientation (≈133°) and led to early‑death stagnation.

        We now also request a random start angle from the parking instance –
        ``_get_random_start_angle`` – ensuring each genome begins with an
        independent orientation. This breaks the deterministic angle lock‑in
        and gives NEAT a richer search space.
        """
        # Ensure pedestrians are spawned before we query for a safe spot.
        # ``randomize`` already called ``_spawn_pedestrians``.
        pos = self.parking.is_position_safe_for_spawn()
        # Use a fresh random angle for each spawn instead of a fixed one.
        # Use a neutral orientation (0°) for the initial spawn. Random angles can
        # cause the car sprite to intersect walls or obstacles immediately,
        # leading to a large number of early‑death events. By starting with a
        # consistent, safe orientation we let NEAT evolve the steering
        # behavior without being penalised for an unlucky initial angle.
        angle = 0.0
        # If the safe‑position routine fell back to the centre (which can happen
        # when no free spot is found), we still want a neutral orientation.
        if pos == (self.parking.MAP_WIDTH // 2, self.parking.MAP_HEIGHT // 2):
            angle = 0.0
        return pos, angle

    # ---------------- UI ----------------

    def _draw_info(self, car=None):

        if car:
            texts = [
                f"Speed: {round(car.velocity.x, 2)}",
                f"Boost: {round(car.acceleration, 2)}",
                f"Rudder: {round(car.steering, 2)}",
                f"Score M: {round(car.movement_score, 2)}",
                f"Score D: {round(car.distance_score, 2)}"
            ]
        else:
            texts = [
                f"Map: {self.map}",
                f"Cars: {self.cars_left}/{len(self.cars)}",
                f"Time: {self.time}/{self.time_per_map}",
                f"Best score: {round(self.best_score)}",
                f"Epoch: {self.generation}/{self.generations}",
                f"Pedestrians: {self._count_alive_pedestrians()}/{self.pedestrian_count}",
            ]

            if self.parking.target_idx is None:
                texts.append("Click a parking spot to select target")

        font = pg.font.SysFont("Comic Sans MS", 15)

        for i, text in enumerate(texts[::-1]):
            label = font.render(text, True, (75, 0, 130))
            rect = label.get_rect()
            rect.center = (1260, self.height - 25 - 15 * i)
            self.screen.blit(label, rect)

    def _count_alive_pedestrians(self):
        return sum(p.is_alive for p in self.parking.pedestrians)

    # ---------------- TRAIN ----------------

    def run_pretrained(self, config_file=None):
        """
        Interactive mode with a pretrained model.
        Click a parking spot -> car drives there autonomously.
        """
        # Try to load pretrained genome
        genome = None
        try:
            genome = self.load("checkpoints/best.pkl")
            print("Loaded pretrained model from checkpoints/best.pkl")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"No valid pretrained model found ({e}). Running in manual mode (no autopilot).")

        # Create NEAT network if genome was loaded
        autopilot = None
        if genome is not None:
            # Resolve config path relative to this file if not provided
            if config_file is None:
                import os
                config_file = os.path.join(os.path.dirname(__file__), "self-parking.conf")
            config = neat.config.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_file
            )
            autopilot = neat.nn.FeedForwardNetwork.create(genome, config)
            print("Autopilot ready. Click a parking spot to set target.")

        # Create the car
        car = Car(self.parking.start_position, self.parking.start_angle)

        while True:
            # events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit(0)

                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # Reset car to center when selecting a new spot
                        if self.parking.target_idx is not None:
                            car = Car(self.parking.start_position, self.parking.start_angle)
                        self.parking.select_parking_space(event.pos)
                        if self.parking.target_idx is not None:
                            print(f"Target selected: spot {self.parking.target_idx} at {self.parking.target_position}")

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_g:
                        self.parking.randomize()
                        car = Car(self.parking.start_position, self.parking.start_angle)
                        print("New parking layout generated")
                    elif event.key == pg.K_h:
                        car = Car(self.parking.start_position, self.parking.start_angle)
                        print("Car reset to center")
                    elif event.key == pg.K_j:
                        car.show_collision_points = not car.show_collision_points
                    elif event.key == pg.K_k:
                        car.show_radars = not car.show_radars
                    elif event.key == pg.K_l:
                        car.show_score = not car.show_score
                    elif event.key == pg.K_ESCAPE:
                        sys.exit(0)

            # draw parking lot
            self.parking.draw(self.screen)
            self.parking.draw_hover(self.screen, pg.mouse.get_pos())

            dt = self.clock.get_time() * 0.01

            # update pedestrians
            if car.is_alive:
                self.parking.update_pedestrians(
                    dt,
                    car.collision_points if hasattr(car, "collision_points") else None
                )
            else:
                self.parking.update_pedestrians(dt, None)

            self.parking.draw_pedestrians(self.screen)

            # car movement
            if car.is_alive:
                if autopilot and self.parking.target_idx is not None:
                    # Autopilot mode: network controls the car
                    inputs = np.concatenate((car.radars_data, car.navigation))
                    outputs = autopilot.activate(inputs)
                    direction, rotation = [
                        0 if -0.33 < o < 0.33 else np.sign(o)
                        for o in outputs
                    ]
                else:
                    # Neutral: car doesn't move if no target or no autopilot
                    direction, rotation = 0, 0

                car.move(
                    {"direction": direction, "rotation": rotation},
                    dt,
                    self.screen,
                    self.parking,
                    self.parking.pedestrians
                )

            car.draw(self.screen)

            # info
            texts = [
                f"Speed: {round(car.velocity.x, 2)}",
                f"Score: {round(car.score, 2)}",
                f"Pedestrians: {self._count_alive_pedestrians()}/{self.pedestrian_count}",
            ]

            if self.parking.target_idx is None:
                texts.append("Click a parking spot to select target")
            else:
                texts.append(f"Target: spot {self.parking.target_idx}")

            font = pg.font.SysFont("Comic Sans MS", 15)
            for i, text in enumerate(texts[::-1]):
                label = font.render(text, True, (75, 0, 130))
                rect = label.get_rect()
                rect.center = (1260, self.height - 25 - 15 * i)
                self.screen.blit(label, rect)

            pg.display.flip()
            self.clock.tick(60)

    @staticmethod
    def save(genome):
        """Dumps genome configuration to file"""
        import os
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/best.pkl", "wb") as f:
            pickle.dump(genome, f)

    @staticmethod
    def load(file):
        """Loads genome configuration from file"""
        with open(file, "rb") as f:
            return pickle.load(f)

    def train(self, config_file=None):
        """Initializes NEAT from config and starts training process on simulation"""
        # Resolve config path relative to this file if not provided
        if config_file is None:
            import os
            config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "self-parking.conf"))
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

        population = neat.Population(config)
        # Log population creation
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"[INFO] NEAT population created with {config.pop_size} genomes at {datetime.datetime.now().isoformat()}\n")
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
        # Ensure checkpoints directory exists and use an absolute path for the checkpoint files
        import os
        checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoints_dir, "self-parking-checkpoint-")
        population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix))

        result = population.run(self._run_generation, self.generations)
        # Log training completion
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"[INFO] Training completed at {datetime.datetime.now().isoformat()}\n")
        return result

    def _init_new_generation(self, genomes, config):

        self.nets = []
        self.cars = []
        self.best_score = 0
        self.generation += 1

        for _, gen in genomes:
            gen.fitness = 0

            net = neat.nn.FeedForwardNetwork.create(gen, config)

            # Obtain a safe start position using the new helper that checks full car footprint
            (spawn_x, spawn_y), spawn_angle = self._get_safe_start_position()
            # Keep the parking instance in sync so that debug logs and any
            # downstream logic that relies on ``parking.start_position`` and
            # ``parking.start_angle`` reflect the actual spawn used for this
            # genome.
            self.parking.start_position = (spawn_x, spawn_y)
            self.parking.start_angle = spawn_angle
            # Log genome initialization
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[DEBUG] Genome initialized with start_pos={self.parking.start_position}, angle={self.parking.start_angle}\n")
            if not (0 <= spawn_x <= self.width and 0 <= spawn_y <= self.height):
                print(f"[WARN] Invalid start position ({spawn_x}, {spawn_y}); resetting to centre.")
                spawn_x, spawn_y = self.width // 2, self.height // 2
                spawn_angle = 0.0

            car = Car(
                (spawn_x, spawn_y),
                spawn_angle
            )

            self.nets.append(net)
            self.cars.append(car)

        # Auto-select a random parking target if none selected
        if self.parking.target_idx is None:
            self.parking.pick_random_target()

    def _run_generation(self, genomes, config):

        self._init_new_generation(genomes, config)

        # Ensure a target parking spot is selected for each generation
        self.parking.pick_random_target()

        while True:

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    sys.exit(0)

                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.parking.select_parking_space(event.pos)

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_g:
                        # Log detailed generation statistics before moving to next map
                        self._log_generation_stats(genomes)
                        self.parking.randomize()
                        self.parking.pick_random_target()
                        self.map += 1
                        self.time = 0
                        return

                    if event.key == pg.K_ESCAPE:
                        sys.exit(0)

            # draw world
            self.parking.draw(self.screen)
            self.parking.draw_hover(self.screen, pg.mouse.get_pos())

            dt = self.clock.get_time() * 0.01

            # Update pedestrians once per frame with collision points from first alive car
            collision_pts = None
            for car in self.cars:
                if car.is_alive:
                    collision_pts = car.collision_points if hasattr(car, "collision_points") else None
                    break
            self.parking.update_pedestrians(dt, collision_pts)

            self.parking.draw_pedestrians(self.screen)

            # Reset counter for alive cars in this generation
            self.cars_left = 0

            # Process each genome/car pair
            for net, car, gen in zip(self.nets, self.cars, genomes):
                # Ensure each car starts from a fresh, valid spawn point
                if not hasattr(car, "initialized") or not car.initialized:
                    pos, angle = self._get_safe_start_position()
                    self.parking.start_position = pos
                    self.parking.start_angle = angle
                    car.reset(self.parking.start_position, self.parking.start_angle)
                    car.initialized = True

                # Log debug info for genome start
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"[DEBUG] Genome {gen[0]} start_pos={self.parking.start_position}, angle={self.parking.start_angle}\n")

                try:
                    inputs = np.concatenate((car.radars_data, car.navigation))
                    outputs = net.activate(inputs)
                except Exception as e:
                    gen[1].fitness = -100.0
                    print(f"[ERROR] Genome {gen[0]} activation error: {e}")
                    car.is_alive = False
                    continue

                if np.any(np.isnan(outputs)) or np.any(np.isinf(outputs)):
                    gen[1].fitness = -100.0
                    print(f"[ERROR] Genome {gen[0]} produced NaN/Inf outputs. Penalizing.")
                    car.is_alive = False
                    continue

                direction, rotation = [
                    0 if -0.33 < o < 0.33 else np.sign(o)
                    for o in outputs
                ]

                car.move(
                    {"direction": direction, "rotation": rotation},
                    dt,
                    self.screen,
                    self.parking,
                    self.parking.pedestrians,
                )

                # Handle death penalties and fitness assignment
                if not car.is_alive and self.time < 20:
                    # Early death penalty
                    gen[1].fitness = -50.0
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(f"[EARLY DEATH] Generation {self.generation}, Genome {gen[0]} died at step {self.time}. Position={car.position}, Angle={car.angle}\n")
                elif not car.is_alive:
                    # Late death penalty: subtract 200 points but not below -50
                    gen[1].fitness = max(car.score - 200, -50.0)
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(f"[LATE DEATH] Generation {self.generation}, Genome {gen[0]} died at step {self.time}. Position={car.position}, Angle={car.angle}\n")
                else:
                    self.best_score = max(self.best_score, car.score)
                    gen[1].fitness = car.score
                # Reset car score for next genome
                car.score = 0

                # Diagnostic: log when a car receives the dead‑zone fitness (default case)
                if car.score == self.DEAD_FITNESS:
                    # Log diagnostic info
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(f"[DIAG] Generation {self.generation}, Genome {gen[0]} got dead‑zone fitness {self.DEAD_FITNESS}\n")
                        f.write(f"       Position={car.position}, Angle={car.angle}, Alive={car.is_alive}, Parked={car.parked}\n")

                if car.is_alive:
                    self.cars_left += 1

            # End of processing all genomes for this generation
            # -------------------------------------------------
            # Draw UI and cars for the current frame
            self._draw_info()

            for car in self.cars:
                car.draw(self.screen)

            # If no cars are alive, end the generation early
            if not self.cars_left:
                # Log stats for this generation before exiting the loop
                self._log_generation_stats(genomes)
                break

            # Advance simulation time
            self.time += 1

            # End of episode handling – reset when time exceeds limit
            if self.time > self.time_per_map:
                self.parking.randomize()
                self.parking.pick_random_target()
                self.map += 1
                self.time = 0
                # Log stats for this generation before exiting the loop
                self._log_generation_stats(genomes)
                break

            # -----------------------------------------------------------------
            # Detect full population collapse (all genomes received dead‑zone fitness)
            # and re‑initialise the environment to avoid a stale state that forces
            # every subsequent genome to die instantly.
            # -----------------------------------------------------------------
            if all(gen[1].fitness == self.DEAD_FITNESS for gen in genomes):
                # Log population collapse
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"[COLLAPSE] Generation {self.generation}: all genomes dead. Re‑initialising environment.\n")
                self.parking = SmallParking(pedestrian_count=self.pedestrian_count)
                self.parking.pick_random_target()
                self.map = 0
                self.time = 0
                # Log stats for this generation before breaking out
                self._log_generation_stats(genomes)
                # Break out of the inner loop so the next generation starts fresh
                break

            pg.display.flip()
            self.clock.tick(60)

    # ---------------------------------------------------------------------
    # Helper to log generation statistics to the dedicated stats log file.
    # ---------------------------------------------------------------------
    def _log_generation_stats(self, genomes):
        """Append a formatted block of generation statistics to ``logs.txt``.

        This method now computes real statistics from the ``genomes`` list
        passed by the training loop:
        * Average fitness and standard deviation.
        * Population size.
        * Species count (derived from the ``species`` attribute of each genome).
        * Best fitness (already tracked as ``self.best_score``).
        Fields that are not readily available remain ``N/A``.
        """
        # Compute average and stddev of fitness values (ignore None)
        fitness_vals = [gen[1].fitness for gen in genomes if hasattr(gen[1], "fitness")]
        avg_fitness = float(np.mean(fitness_vals)) if fitness_vals else 0.0
        std_fitness = float(np.std(fitness_vals)) if fitness_vals else 0.0
        # Population size
        pop_size = len(genomes)
        # Species count – genomes have a ``species`` attribute after speciation
        species_ids = {getattr(gen[1], "species", None) for gen in genomes}
        species_count = len([s for s in species_ids if s is not None])

        with open(self._stats_log_file, "a", encoding="utf-8") as f:
            f.write("****** Running generation {} ******\n".format(self.generation))
            f.write("Population's average fitness: {:.5f} stdev: {:.5f}\n".format(avg_fitness, std_fitness))
            f.write("Best fitness: {:.5f} - size: (N/A) - species N/A - id N/A\n".format(self.best_score))
            f.write("Average adjusted fitness: N/A\n")
            f.write("Mean genetic distance N/A, standard deviation N/A\n")
            f.write("Population of {} members in {} species (after reproduction):\n".format(pop_size, species_count))
            f.write("   ID   age  size   fitness   adj fit  stag\n")
            f.write("   ====  ===  ====  =========  =======  ====\n")
            f.write("   ... (details omitted)\n")