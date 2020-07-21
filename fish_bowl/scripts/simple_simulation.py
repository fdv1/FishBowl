import logging
import argparse
import time

from fish_bowl.dataio.persistence import SimulationClient, get_database_string
from fish_bowl.process.base import SimulationGrid
from fish_bowl.common.config_reader import read_simulation_config
from fish_bowl.process.simple_display import display_simple_grid

_logger = logging.getLogger(__name__)


def adjacent_locations(speed=1):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != j:
                yield i * speed, j * speed

from itertools import chain, repeat
from random import choice, shuffle, random
from typing import List

ADJACENT_CELLS = tuple(adjacent_locations(1))


def possible_locations(grid, x, y, moves):
    for x_move, y_move in moves:
        x_new, y_new = x + x_move, y + y_move
        if len(grid) > x_new > 0 and len(grid) > y_new > 0:
            yield x_new, y_new


class Animal:

    def __init__(self, grid, tob):
        self.tob = tob  # turn of birth
        self.x = 0
        self.y = 0
        self.grid = grid
        self.POPULATION.add(self)
        self.alive = True
        self.NB_BIRTHS += 1

    def update_liveliness(self, _turn):
        pass

    def set_location(self, x, y):
        if self.grid[x][y]:
            return False
        self.grid[self.x][self.y] = None
        self.x = x
        self.y = y
        self.grid[self.x][self.y] = self
        return True

    def dies(self):
        assert self.alive
        self.alive = False
        self.grid[self.x][self.y] = None
        self.__class__.POPULATION.remove(self)

    def move(self):
        shuffle(self.MOVES)
        for x, y in possible_locations(self.grid, self.x, self.y, self.MOVES):
            if self.set_location(x, y):
                return True

    def breed(self, turn, x, y):
        if turn > self.tob + self.BREED_MATURITY and random() <= self.BREED_PROBABILITY:
            self.__class__(self.grid, turn).set_location(x, y)

    def evolve(self, turn):
        self.update_liveliness(turn)
        if self.alive:
            x, y = self.x, self.y
            # can only breed if it has moved
            if self.eat(turn) or self.move():
                self.breed(turn, x, y)

    def eat(self, _turn):
        return False


class Fish(Animal):
    MOVES = []
    POPULATION = set()
    BREED_PROBABILITY = 0
    BREED_MATURITY = 3
    SPEED = 0
    NB_BIRTHS = 0
    NB_DEATHS = 0


class Shark(Animal):
    MOVES = []
    POPULATION = set()
    BREED_PROBABILITY = 80
    BREED_MATURITY = 1
    SPEED = 0
    STARVE_AFTER = 10
    NB_BIRTHS = 0
    NB_DEATHS = 0

    def __init__(self, *args):
        super().__init__(*args)
        self.last_meal = self.tob

    def update_liveliness(self, turn):
        if turn - self.last_meal > self.STARVE_AFTER:
            self.dies()

    def eat(self, turn):
        for x, y in possible_locations(self.grid, self.x, self.y, ADJACENT_CELLS):
            maybe_prey = self.grid[x][y]
            if isinstance(maybe_prey, Fish):
                maybe_prey.dies()
                self.last_meal = turn
                assert self.set_location(x, y), "internal error"
                assert self.grid[x][y] is self
                return True


def euclidian_division(x, y):
    return x // y, x % y

def simulate_fishbowl(grid_size,
                      nb_turns,
                      init_nb_fish,
                      init_nb_shark,
                      fish_breed_maturity,
                      shark_breed_maturity,
                      fish_breed_probability,
                      shark_breed_probability,
                      fish_speed,
                      shark_speed,
                      shark_starving,
                      ):
    assert grid_size * grid_size > (init_nb_fish + init_nb_shark), "fishbowl too packed...!"
    grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    indexes = list(range(grid_size))
    Fish.BREED_MATURITY = fish_breed_maturity
    Shark.BREED_MATURITY = shark_breed_maturity
    Fish.BREED_PROBABILITY = fish_breed_probability / 100.0
    Shark.BREED_PROBABILITY = shark_breed_probability / 100.0
    Fish.MOVES = list(adjacent_locations(fish_speed))
    Shark.MOVES = list(adjacent_locations(shark_speed))
    Shark.STARVE_AFTER = shark_starving

    for AnimalClass, initial_nb in [(Fish, init_nb_fish), (Shark, init_nb_shark)]:
        nb_animals_per_year, remaining = euclidian_division(initial_nb, AnimalClass.BREED_MATURITY)
        for age in chain(chain.from_iterable(repeat(-age, nb_animals_per_year)
                                             for age in range(AnimalClass.BREED_MATURITY)),
                         repeat(0, remaining)):
            animal = AnimalClass(grid, age)
            while True:
                x, y = choice(indexes), choice(indexes)
                if animal.set_location(x, y):
                    break
    turn = 0
    while turn < nb_turns:
        for animal in chain(list(Shark.POPULATION), list(Fish.POPULATION)):
            animal.evolve(turn)
        if not Shark.POPULATION:
            break
        # print(turn, len(Fish.POPULATION), len(Shark.POPULATION))
        turn += 1
    return turn, Fish.POPULATION, Shark.POPULATION


class Timer:
    def __init__(self):
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start


if __name__ == '__main__':
    nb_turns = 100
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d:%(message)s")
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('--config_name', default='simulation_config_1',
                            help='Simulation configuration file name')
    cmd_parser.add_argument('--max_turn', default=100, type=int, help='Maximum number of turns for the simulation')
    cmd_parser.add_argument('--config_path', default=None, type=str,
                            help="""
                            Configuration file path. If specified, configuration file will be loaded from this path
                            """)
    args = cmd_parser.parse_args()
    if args.config_path is not None:
        raise NotImplementedError('Code for directing to an alternative configuration'
                                  ' repository has not been implemented yet')
    # Load simulation configuration
    sim_config = read_simulation_config(args.config_name)
    with Timer() as timer:
        n, x, y = simulate_fishbowl(nb_turns=nb_turns, **sim_config)
    print(timer.elapsed, n, len(x), len(y))
    # Instantiate client
    client = SimulationClient(get_database_string())
    # display initial grid
    grid = SimulationGrid(persistence=client, simulation_parameters=sim_config)
    print(display_simple_grid(client.get_animals_df(grid._sid), grid_size=sim_config['grid_size']))
    for turn in range(args.max_turn):
        timer = time.time()
        grid.play_turn()
        print(''.join(['*'] * sim_config['grid_size'] * 2))
        print('Turn: {turn: ^{size}}'.format(turn=grid._sim_turn, size=sim_config['grid_size']))
        print()
        print(grid.get_simulation_grid_data())
        print("======")
        # print(display_simple_grid(grid.get_simulation_grid_data(), sim_config['grid_size']))
        print()
        print('Turn duration: {:<3}s'.format(int(time.time()-timer)))
        print()
