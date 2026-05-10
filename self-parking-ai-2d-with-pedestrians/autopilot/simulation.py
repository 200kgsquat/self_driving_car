import os
# Use dummy video driver for headless training
os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import neat
import pickle
import numpy as np
import pygame as pg

from autopilot.car import Car
from autopilot.parking import SmallParking


class Simulation:
    """
    Self-driving parking simulation with interactive selection
    """

    def __init__(
        self,
        epochs=10000,
        parked_cars=None,
        time_per_map=1000,
        pedestrian_count=5
    ):
        pg.init()

        pg.display.set_caption(
            "Self-parking simulation with pedestrians"
        )

        self.window = (1320, 768)
        self.width, self.height = self.window

        self.screen = pg.display.set_mode(self.window)

        self.clock = pg.time.Clock()

        self.parking = SmallParking(
            pedestrian_count=pedestrian_count
        )

        self.best_score = -float("inf")
        # Ensure a target is selected right from the start
        self.parking.pick_random_target()

        self.time_per_map = time_per_map
        self.generations = epochs

        self.generation = 0
        self.map = 0
        self.time = 0

        self.cars_left = 0
        self.pedestrian_count = pedestrian_count

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

    def run_pretrained(self, config_file="autopilot/self-parking.conf"):
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

    def train(self, config_file="autopilot/self-parking.conf"):
        """Initializes NEAT from config and starts training process on simulation"""
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
        population.add_reporter(neat.Checkpointer(10, filename_prefix="checkpoints/self-parking-checkpoint-"))

        return population.run(self._run_generation, self.generations)

    def _init_new_generation(self, genomes, config):

        self.nets = []
        self.cars = []
        self.best_score = 0
        self.generation += 1

        for _, gen in genomes:
            gen.fitness = 0

            net = neat.nn.FeedForwardNetwork.create(gen, config)

            car = Car(
                self.parking.start_position,
                self.parking.start_angle
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

            self.cars_left = 0

            for net, car, gen in zip(self.nets, self.cars, genomes):

                inputs = np.concatenate((car.radars_data, car.navigation))
                outputs = net.activate(inputs)

                direction, rotation = [
                    0 if -0.33 < o < 0.33 else np.sign(o)
                    for o in outputs
                ]

                car.move(
                    {"direction": direction, "rotation": rotation},
                    dt,
                    self.screen,
                    self.parking,
                    self.parking.pedestrians
                )

                self.best_score = max(self.best_score, car.score)
                gen[1].fitness = car.score

                if car.is_alive:
                    self.cars_left += 1

            self._draw_info()

            for car in self.cars:
                car.draw(self.screen)

            if not self.cars_left:
                break

            self.time += 1

            if self.time > self.time_per_map:
                self.parking.randomize()
                self.parking.pick_random_target()
                self.map += 1
                self.time = 0
                break

            pg.display.flip()
            self.clock.tick(60)