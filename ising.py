import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class Ising:
    def __init__(self, size, beta, max_unchanging=10):
        self.size = size
        self.beta = beta
        self.board = None
        self.cells = [(i, j) for j in range(size) for i in range(size)]
        self.max_unchanging = max_unchanging
        self.fig = plt.figure()
        self.im = None

    def init_board(self, mode):
        if mode == "random":
            self.board = 2 * np.random.randint(2, size=(self.size, self.size)) - 1
        elif mode == "ones":
            self.board = np.ones((self.size, self.size))
        elif mode == "zeros":
            self.board = np.zeros((self.size, self.size))

    def probability(self, energy):
        return 1 / (1 + np.exp(2 * energy * self.beta))

    def get_order(self):
        return np.random.permutation(self.cells)

    def get_energy(self, x, y):
        return 2 * self.board[(x, y)] * sum(
            self.board[((x + dx) % self.size, (y + dy) % self.size)] for dx, dy in DIRECTIONS)

    def step(self):
        changed = False
        total_energy = 0
        order = self.get_order()
        for x, y in order:
            energy = self.get_energy(x, y)
            p = np.random.rand()
            if energy < 0 or p < self.probability(energy):
                self.board[(x, y)] *= -1
                changed = True
            total_energy -= energy / 2
        return changed, total_energy

    def simulate(self, n_steps, render=False):
        avg_energy = 0
        avg_squared_energy = 0
        unchanged = 0
        fig = plt.figure()
        for _ in range(n_steps):
            changed, energy = self.step()
            avg_energy += energy
            avg_squared_energy += energy * energy

            if not changed:
                unchanged += 1
                if unchanged >= self.max_unchanging:
                    break
            else:
                unchanged = 0
        capacity = self.beta * self.beta * (avg_squared_energy - avg_energy * avg_energy / n_steps) / n_steps
        return capacity

    def stabilize(self, n_steps):
        for _ in range(n_steps):
            self.step()

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


#
#
#
# for _ in range(pre_steps):
#     step()

if __name__ == "__main__":
    ising = Ising(8, 0.4)
    ising.init_board("random")
    # ising.stabilize(1000)
    # ising.simulate(10000)
    # fig = plt.figure()
    ising.animate()
    plt.show()
