from autopilot import Simulation


def train():
    """Train a new model"""
    # Reduced time_per_map to a more reasonable 300 seconds
    # Give the car a bit more time to reach the target with higher speed limits
    sim = Simulation(epochs=1000, time_per_map=500, pedestrian_count=3)
    best_genome = sim.train()
    sim.save(best_genome)
    print("Training complete! Model saved to checkpoints/best.pkl")


def test():
    """Run the pretrained model with interactive parking spot selection"""
    sim = Simulation(epochs=1000, time_per_map=200, pedestrian_count=3)
    sim.run_pretrained()


if __name__ == '__main__':
    # Train a new model:
    train()

    # Run pretrained model (interactive parking spot selection):
    # test()
