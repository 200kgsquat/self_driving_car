# Self-Parking AI 2D with Pedestrians

This project is an extension of the original self-parking AI simulation that adds **pedestrians** walking around the parking lot. The car must learn to navigate and park while avoiding collisions with pedestrians.

## Features

- **Self-driving car simulation** using NEAT (NeuroEvolution of Augmenting Topologies)
- **Pedestrians** that walk randomly around the parking lot
- **Collision detection** between car and pedestrians
- **Radar system** that detects both static obstacles and moving pedestrians
- **Training mode** - evolve neural network to park the car autonomously
- **Test mode** - test trained model or manually control the car

## Installation

1. Install Python 3.7+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
```bash
python main.py
```

This will train the autopilot using NEAT algorithm. The training process will:
- Generate populations of neural networks
- Test each network by controlling a car in the simulation
- Select the best performing networks for the next generation
- Save the best genome to `checkpoints/best.pkl`

### Controls (in test mode without autopilot)
- **W/↑** - Accelerate forward
- **S/↓** - Accelerate backward
- **A/←** - Steer left
- **D/→** - Steer right
- **G** - Generate new parking layout
- **H** - Reset car position
- **J** - Show/hide collision points
- **K** - Show/hide radar lines
- **L** - Show/hide score
- **ESC** - Exit

## Project Structure

```
self-parking-ai-2d-with-pedestrians/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── autopilot/
│   ├── __init__.py
│   ├── car.py             # Car physics and sensors
│   ├── parking.py         # Parking lot with pedestrians
│   ├── pedestrian.py      # Pedestrian behavior
│   ├── simulation.py      # Main simulation loop
│   ├── self-parking.conf  # NEAT configuration
│   └── sprites/           # Images for cars and environment
├── checkpoints/           # Saved neural network checkpoints
└── demo/                  # Demo videos and images
```

## How Pedestrians Work

1. **Spawning**: Pedestrians spawn at random positions on the road when the map is generated
2. **Movement**: They walk in random directions, pause, then choose new directions
3. **Collision**: If a car hits a pedestrian:
   - The car receives a heavy penalty (-100 points)
   - The car stops immediately
   - The pedestrian is marked as "hit"
4. **Detection**: The car's radar system detects pedestrians and includes them in distance calculations

## Scoring

- **Distance score**: Based on how close the car gets to the target parking spot
- **Movement score**: Penalizes unnecessary movement and collisions
- **Pedestrian collision**: -100 points (major penalty)
- **Obstacle collision**: -10 points

## Configuration

Edit `autopilot/self-parking.conf` to modify NEAT parameters:
- `pop_size` - Number of cars per generation
- `num_hidden` - Hidden layers in neural network
- `fitness_threshold` - Score needed to complete training

Edit `main.py` to modify simulation parameters:
- `epochs` - Number of generations to train
- `time_per_map` - Time limit per parking layout
- `pedestrian_count` - Number of pedestrians on the map

## License

See the original project's LICENSE file.