from autopilot import Simulation


def train():
    """Train a new model"""
    # Reduced time_per_map to a more reasonable 300 seconds
    # Give the car a bit more time to reach the target with higher speed limits
    # Enable GUI during training by setting show_gui=True
    # Updated time_per_map to match the new default limit of 250 steps
    sim = Simulation(epochs=1000, time_per_map=250, pedestrian_count=3, show_gui=True)
    best_genome = sim.train()
    sim.save(best_genome)
    print("Training complete! Model saved to checkpoints/best.pkl")


def test():
    """Run the pretrained model with interactive parking spot selection"""
    # For testing (pretrained model) we also want the GUI visible
    # Updated time_per_map to match the new default limit of 250 steps (or use default)
    sim = Simulation(epochs=1000, time_per_map=250, pedestrian_count=3, show_gui=True)
    sim.run_pretrained()


if __name__ == '__main__':
    # Train a new model:
    train()

    # Run pretrained model (interactive parking spot selection):
    # test()
