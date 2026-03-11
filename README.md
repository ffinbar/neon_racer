# Neon Racer


## Requirements
- Python 3.8+

## Install

You can use VSCode’s built in cloning GUI, or:
```bash
git clone [repository-url] neon_racer_course
```
Once you have the repository folder open, navigate to the `neon_racer` folder:
```bash
fin@Fins-MacBook-Air neon_racer_course % cd neon_racer
fin@Fins-MacBook-Air neon_racer %
```
Create virtual environment:
```bash
fin@Fins-MacBook-Air neon_racer % python3 -m venv venv
fin@Fins-MacBook-Air neon_racer % source venv/bin/activate
# Windows: venv\Scripts\activate
(venv) fin@Fins-MacBook-Air neon_racer %
```

You will see `(venv)` next to your terminal which tells you that it is activated and working properly.
Now you are ready to install dependencies:
```bash
(venv) fin@Fins-MacBook-Air neon_racer % pip install -r requirements.txt
```
This should start to download a long list of files which the program(s) in this course will use. It is important that we are using the (venv) because otherwise Python will try to install these dependencies globally for our user, which can create issues and make a mess. It is best to contain these files to our workspace.

**It is very important that all students get their virtual environment working before starting the course.**

## Controls

| Key       | Action                                                                 |
|-----------|------------------------------------------------------------------------|
| Arrow Keys| Manual driving                                                         |
| Space     | Brake                                                                  |
| V         | Toggles headless mode when training. Very useful!                      |
| R         | Restart race                                                           |
| C         | Toggle camera view                                                     |
| D         | Toggle debug overlay (sensors)                                         |
| H         | Toggle HUD                                                             |
| P         | Pause                                                                  |
| [ / ]     | Decrease / Increase simulation speed                                   |

### A note on training speeds
When training models, pressing `V` will toggle headless mode, which stops rendering the game window. This allows the simulation to run as fast as possible. **This is your best friend when training agents.** It allows you to check in on progress and then resume training at faster-than-real-time speeds.

When training models with multiple parallel jobs (details below), the game window will not be displayed at all and the same effect is achieved.

## Running the game
```
usage: python -m neon_racer.main [-h] [--track TRACK] [--agents AGENTS [AGENTS ...]] [--no-sound]
```
Running any of the following commands with -h is a good way to see all available options.

### Play manually
The manual agent allows you to drive the car using your keyboard. It is chosen by default if no agents are specified.
```
python -m neon_racer.main --track default --agents manual
```

### Picking a track
You can choose from these built-in tracks: `default`, `level2`, and `level3`.
```
python -m neon_racer.main --track level2 --agents manual
```

### Run an agent
There are three built-in agents: `rover`, `neat`, and `rl`.
```
python -m neon_racer.main --track default --agents rover
python -m neon_racer.main --track default --agents neat
python -m neon_racer.main --track default --agents rl
```

### Multi-agent race
Multiple agents can be specified by separating them with spaces. Use `manual` to include a human player.
```
python -m neon_racer.main --agents rover neat rl manual
```
### Specify model paths
Specific models can be loaded by appending the model name after a colon. They are loaded from the `models/` directory for RL models and `genomes/` directory for NEAT genomes.
```
python -m neon_racer.main --agents neat:genome_name rl:model_name
```
## Train NEAT
```
usage: python -m neon_racer.train_neat [-h] [--gens GENS] [--track TRACK] [--speed SPEED] [--checkpoint CHECKPOINT] [--jobs JOBS] [--cpenable] [--dashboard] [--no-dashboard] [--no-sound] [--measure] [--measure-file MEASURE_FILE]
```
Trains a NEAT agent for 50 generations on the default track.
```
python -m neon_racer.train_neat --track default --gens 50
```
### Train NEAT and save checkpoint
When training finishes, saves a checkpoint file to `checkpoints/` directory. This can be used to resume training later.
```
python -m neon_racer.train_neat --track default --gens 50 --cpenable
```
### Train NEAT from checkpoint
Resumes NEAT training from a previously saved checkpoint file.
```
python -m neon_racer.train_neat --track default --checkpoint neat-checkpoint-xyz
```
## Train RL
```
usage: python -m neon_racer.train_rl [-h] [--track TRACK] [--timesteps TIMESTEPS] [--load LOAD] [--save SAVE] [--jobs JOBS] [--dashboard] [--no-dashboard] [--no-sound]
```
Trains an RL agent for 500,000 timesteps on the default track.
```
python -m neon_racer.train_rl --track default --timesteps 500000
```
### Train RL from model
Resumes RL training from a previously saved model.
```
python -m neon_racer.train_rl --track default --timesteps 500000 --load model_name
```
## Train NEAT/RL (multi-threaded)
Runs training using multiple parallel jobs. This will not open a game window, and can significantly speed up training on machines with multiple CPU cores.
As a rule of thumb, set `--jobs` to the number of CPU cores available or less.
When set to 1, training runs in single-threaded mode and will draw the game window (this is the default behaviour).
```
python -m neon_racer.train_neat --track default --gens 200 --jobs 4
python -m neon_racer.train_rl --track default --timesteps 500000 --jobs 4
```
## Custom tracks
Custom tracks can be created by adding new image files to the `student/assets/tracks/` directory.
They should be in PNG format with a single colour value as the background/track boundary.
The rest of the track can use any colours to represent different surfaces. The car will crash when it touches the boundary colour.
To turn the image into a track, run:
```
python -m neon_racer.track_editor --track your_track_image.png
```

Good luck!

---

Fin 2026
