import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import argparse
from scipy.signal import convolve2d

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class Ising:
    def __init__(self, size, beta, initialisation_mode="random"):
        self.size = size
        self.mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        self.beta = beta
        self.board = None
        self.cells = [(i, j) for j in range(size) for i in range(size)]
        self.initialisation_mode = initialisation_mode
        self.fig = plt.figure()
        self.im = None
        self.init_board()

    def init_board(self):
        if self.initialisation_mode == "random":
            self.board = 2 * np.random.randint(2,
                                               size=(self.size, self.size)) - 1
        elif self.initialisation_mode == "ones":
            self.board = np.ones((self.size, self.size))
        elif self.initialisation_mode == "zeros":
            self.board = np.zeros((self.size, self.size))
        elif self.initialisation_mode == "half":
            self.board = np.array(
                [[i > self.size / 2 for i in range(self.size)]
                 for j in range(self.size)])

    def probability(self, energy):
        return 1 / (1 + np.exp(2 * energy * self.beta))

    def get_order(self):
        return np.random.permutation(self.cells)

    def get_energy(self, x, y):
        return 2 * self.board[(x, y)] * sum(self.board[((x + dx) % self.size,
                                                        (y + dy) % self.size)]
                                            for dx, dy in DIRECTIONS)

    def get_total_energy(self):
        H = convolve2d(self.board, self.mask, mode='same', boundary="wrap")
        return np.exp(-self.beta * np.sum(H * self.board))

    def step(self):
        order = self.get_order()
        for x, y in order:
            energy = self.get_energy(x, y)
            p = np.random.rand()
            if energy < 0 or p < self.probability(energy):
                self.board[(x, y)] *= -1

    def simulate(self, n_steps):
        for _ in range(n_steps):
            self.step()
        return self.get_total_energy()

    def set_beta(self, beta):
        self.beta = beta

    def show_state(self):
        plt.imshow(self.board, vmin=-1, vmax=1)
        plt.show()

    def animate(self):
        self.im = plt.imshow(self.board.copy(), animated=True, vmin=-1, vmax=1)
        ani = animation.FuncAnimation(self.fig, self.update_figure, frames=60)
        plt.show()

    def update_figure(self, i):
        self.step()
        self.im.set_array(self.board.copy())
        return self.im,

    def get_capacities(self, betas, stabilisation_steps, simulation_number):
        capacities = []
        for beta in betas:
            self.set_beta(beta)
            self.init_board()

            total_energy = 0
            T1 = 0
            T2 = 0
            T3 = 0

            for _ in range(simulation_number):
                energy = self.simulate(stabilisation_steps)
                Hi = self.get_energy(0, 0)
                ch = np.cosh(beta * Hi)
                x = beta * Hi * np.tanh(beta * Hi) - np.log(2 * ch)
                T1 += Hi * Hi * beta * beta / (ch * ch)
                T2 += -beta * energy * x
                T3 += beta * x
                total_energy += energy

            capacity = (T1 + T2 + T3 * total_energy /
                        simulation_number) / simulation_number
            print(f"beta: {beta} , capacity: {capacity}")
            capacities.append(capacity)
        return capacities


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ising Simulation")
    parser.add_argument("--initialisation_mode", type=str, default="random")
    parser.add_argument("--size", type=int, default=8)
    subparsers = parser.add_subparsers(dest="command")
    animate_parser = subparsers.add_parser("animate")
    animate_parser.add_argument("--beta", type=float, default=0.4)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--stabilisation_steps",
                                 type=int,
                                 default=100,
                                 required=False)
    simulate_parser.add_argument("--simulation_number",
                                 type=int,
                                 default=1000,
                                 required=False)

    args = parser.parse_args()
    assert (args.command and args.command in ["simulate", "animate"])

    size = args.size
    initialisation_mode = args.initialisation_mode
    if args.command == "simulate":
        ising = Ising(size, None, initialisation_mode)
        betas = [10**i for i in np.linspace(-1, 0.3, 14)]
        capacities = ising.get_capacities(betas, args.stabilisation_steps,
                                          args.simulation_number)
        print(capacities)
        plt.xscale('log')
        plt.plot(betas, capacities)
        plt.show()
    else:
        beta = args.beta
        ising = Ising(size, beta, initialisation_mode)
        ising.animate()
