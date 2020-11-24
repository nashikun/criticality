import numpy as np
import argparse

from simulation import Simulation


class Boltzman(Simulation):
    def __init__(self, size, beta, initialisation_mode):
        self.cells = [i for i in range(size)]
        super().__init__(size, beta, initialisation_mode)
        self.init_board((size, 1))
        weight = np.random.random((size, 1))
        self.weights = weight + weight.T

    def get_Hi(self, position):
        return self.weights[position] @ self.board

    def get_order(self):
        return np.random.permutation(self.cells)

    def get_total_energy(self):
        return 0

    def get_delta_energy(self, position):
        print(self.board[position], self.get_Hi(position))
        return 2 * self.board[position] * self.get_Hi(position)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Boltzman Simulation")
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
        boltzman = Boltzman(size, None, initialisation_mode)
        betas = [10**i for i in np.linspace(-1.5, 0.5, 81)]
        plt.xscale('log')
        plt.plot(betas, capacities)
        plt.plot(betas, local_capacities)
        plt.show()
    else:
        beta = args.beta
        ising = Boltzman(size, beta, initialisation_mode)
        ising.animate()
