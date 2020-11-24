import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class Simulation:
    def __init__(self, size, beta, initialisation_mode="random"):
        self.size = size
        self.beta = beta
        self.board = None
        self.initialisation_mode = initialisation_mode
        self.fig = plt.figure()
        self.im = None

    def init_board(self, dimensions):
        if self.initialisation_mode == "random":
            self.board = 2 * np.random.randint(2, size=dimensions) - 1
        elif self.initialisation_mode == "ones":
            self.board = np.ones(dimensions)
        elif self.initialisation_mode == "zeros":
            self.board = np.zeros(dimensions)
        else:
            raise ValueError("Unrecognised initialisation method: " +
                             self.initialisation_mode)

    def probability(self, energy):
        return 1 / (1 + np.exp(2 * energy * self.beta))

    def get_order(self):
        raise NotImplementedError()

    def get_delta_energy(self, position):
        return 2 * self.board[position] * self.get_Hi(position)

    def get_Hi(self, position):
        raise NotImplementedError()

    def get_total_energy(self):
        raise NotImplementedError()

    def step(self):
        order = self.get_order()
        for position in order:
            energy = self.get_delta_energy(position)
            p = np.random.rand()
            if energy < 0 or p < self.probability(energy):
                self.board[position] *= -1

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
        self.im = plt.imshow(self.board.copy(),
                             animated=True,
                             vmin=-1,
                             vmax=1,
                             cmap="gray")
        ani = animation.FuncAnimation(self.fig, self.update_figure, frames=60)
        # ani.save('myAnimation.gif', writer='imagemagick', fps=3)
        plt.show()

    def update_figure(self, i):
        self.step()
        self.im.set_array(self.board.copy())
        return self.im,
