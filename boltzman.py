import numpy as np
import argparse

from simulation import Simulation


class Boltzman(Simulation):
    def __init__(self, input_size, hidden_size, beta, initialisation_mode):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cells = [i for i in range(input_size + hidden_size)]
        super().__init__(input_size + hidden_size, beta, initialisation_mode)
        self.init_board((input_size + hidden_size, 1))
        weight = np.random.random((input_size + hidden_size, 1))
        self.weights = weight + weight.T

    def get_Hi(self, position):
        return self.weights[position] @ self.board

    def get_order(self):
        return np.random.permutation(self.cells[self.input_size:])

    def get_total_energy(self):
        return np.exp(-self.beta * self.board.T @ self.weights @ self.board)

    def get_delta_energy(self, position):
        return 2 * self.board[position] * self.get_Hi(position)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Boltzman Simulation")
    parser.add_argument("--initialisation_mode", type=str, default="random")
    parser.add_argument("--input_size", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=5)
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

    input_size = args.input_size
    hidden_size = args.hidden_size
    initialisation_mode = args.initialisation_mode
    if args.command == "simulate":
        boltzman = Boltzman(input_size, hidden_size, None, initialisation_mode)
        betas = [10**i for i in np.linspace(-1.5, 0.5, 81)]
        plt.xscale('log')
        plt.plot(betas, capacities)
        plt.plot(betas, local_capacities)
        plt.show()
    else:
        beta = args.beta
        ising = Boltzman(input_size, hidden_size, beta, initialisation_mode)
        ising.animate()
