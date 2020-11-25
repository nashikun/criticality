import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import convolve2d

from simulation import Simulation

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MASK = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

class Ising(Simulation):
    def __init__(self, size, beta, initialisation_mode="random"):
        self.cells = [(i, j) for j in range(size) for i in range(size)]
        super().__init__(size, beta, initialisation_mode)
        self.init_board((size, size))

    def get_delta_energy(self, position):
        return 2 * self.board[position] * self.get_Hi(position)

    def get_order(self):
        return [tuple(x) for x in np.random.permutation(self.cells)]

    def get_Hi(self, position):
        x, y = position
        return sum(self.board[((x + dx) % self.size, (y + dy) % self.size)] for dx, dy in DIRECTIONS)

    def get_total_energy(self):
        H = convolve2d(self.board, MASK, mode='same', boundary="wrap")
        return np.exp(-self.beta * np.sum(H * self.board))

    def get_capacities(self, betas, stabilisation_steps, simulation_number):
        capacities = []
        local_capacities = []
        for beta in betas:
            self.set_beta(beta)
            self.init_board((size, size))

            total_energy = 0
            cell_energy = 0
            T1 = 0
            T2 = 0
            T3 = 0
            T4 = 0

            for _ in range(simulation_number):
                energy = self.simulate(stabilisation_steps)
                # self.step()
                # energy = self.get_total_energy()
                Hi = self.get_Hi((0, 0))
                ch = np.cosh(beta * Hi)
                x = beta * Hi * np.tanh(beta * Hi) - np.log(2 * ch)
                T1 += Hi * Hi * beta * beta / (ch * ch)
                T2 += beta * x
                T3 -= beta * x * energy
                T4 -= beta * x * Hi * self.board[(0, 0)]
                total_energy += energy
                cell_energy += Hi * self.board[(0, 0)]
                
            capacity = (T1 + T2 * total_energy / simulation_number + T3) / simulation_number 
            local_capacity = (T1 - T2 * cell_energy / simulation_number - T4) / simulation_number 
            print(f"beta: {beta} , capacity: {capacity}, local_capacity: {local_capacity}")
            capacities.append(capacity)
            local_capacities.append(T1 / simulation_number )
            
        return capacities, local_capacities

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
        betas = [10**i for i in np.linspace(-1.5, 0.5, 21)]
        capacities, local_capacities = ising.get_capacities(betas, args.stabilisation_steps,
                                          args.simulation_number)
        print(capacities)
        plt.xscale('log')
        plt.plot(betas, capacities)
        plt.plot(betas, local_capacities)
        plt.show()
    else:
        beta = args.beta
        ising = Ising(size, beta, initialisation_mode)
        ising.animate()

