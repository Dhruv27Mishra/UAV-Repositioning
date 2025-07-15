
import numpy as np
import matplotlib.pyplot as plt
def jains_fairness(user_rates):
        user_rates = np.array(user_rates)
        numerator = np.sum(user_rates) ** 2
        denominator = len(user_rates) * np.sum(user_rates ** 2)
        return numerator / denominator if denominator != 0 else 0
class DeterministicUAVSimulation:
    
    def __init__(self, num_uavs=3, num_users=20, grid_size=(10, 10), uav_altitude=12.0, tx_power=1.0, noise_power=1e-3, path_loss_exp=2.5, bandwidth=1.0):
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.grid_size = grid_size
        self.uav_altitude = uav_altitude
        self.tx_power = tx_power
        self.noise_power = noise_power
        self.path_loss_exp = path_loss_exp
        self.bandwidth = bandwidth

        self.uav_positions = self._initialize_uav_positions()
        self.user_positions = self._initialize_user_positions()
        self.association = np.zeros(self.num_users, dtype=int)
        self.user_rates = np.zeros(self.num_users)

    def _initialize_uav_positions(self):
        x_positions = np.linspace(2, self.grid_size[0] - 2, self.num_uavs)
        y_position = self.grid_size[1] / 2
        z_position = self.uav_altitude
        return np.array([[x, y_position, z_position] for x in x_positions])

    def _initialize_user_positions(self):
        positions = np.random.rand(self.num_users, 3)
        positions[:, 0] *= self.grid_size[0]
        positions[:, 1] *= self.grid_size[1]
        positions[:, 2] = 0
        return positions

    def _compute_sinr(self):
        fading = np.random.exponential(1.0, size=(self.num_uavs, self.num_users))
        sinr_matrix = np.zeros((self.num_uavs, self.num_users))

        for i in range(self.num_uavs):
            for j in range(self.num_users):
                d_ij = np.linalg.norm(self.uav_positions[i] - self.user_positions[j]) + 1e-3
                signal = self.tx_power * fading[i, j] / (d_ij ** self.path_loss_exp)
                interference = 0.0
                for k in range(self.num_uavs):
                    if k != i:
                        d_kj = np.linalg.norm(self.uav_positions[k] - self.user_positions[j]) + 1e-3
                        interference += self.tx_power * fading[k, j] / (d_kj ** self.path_loss_exp)
                sinr_matrix[i, j] = signal / (self.noise_power + interference)
        return sinr_matrix

    def run(self):
        sinr_matrix = self._compute_sinr()
        best_uavs = np.argmax(sinr_matrix, axis=0)
        self.association = best_uavs

        total_throughput = 0.0
        for j in range(self.num_users):
            i = best_uavs[j]
            sinr = sinr_matrix[i, j]
            rate = self.bandwidth * np.log2(1 + sinr)
            self.user_rates[j] = rate
            total_throughput += rate
        return total_throughput, self.user_rates

    def visualize(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.user_positions[:, 0], self.user_positions[:, 1], c='green', label='Users')
        plt.scatter(self.uav_positions[:, 0], self.uav_positions[:, 1], c='blue', s=100, label='UAVs')

        for j in range(self.num_users):
            user = self.user_positions[j, :2]
            uav = self.uav_positions[self.association[j], :2]
            plt.plot([user[0], uav[0]], [user[1], uav[1]], 'gray', alpha=0.4)

        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Deterministic UAV Placement and User Associations')
        plt.legend()
        plt.grid(True)
        plt.show()
        


if __name__ == "__main__":
    user_counts = [1,2,3,4,5]
    fairness_scores = []
    sim = DeterministicUAVSimulation()
    total_throughput, user_rates = sim.run()
    print(f"Total Throughput: {total_throughput:.2f}")
    print(f"User Rates: {user_rates}")
    sim.visualize()
    for num_users in user_counts:
        total_fairness = 0
        runs = 1000
        for _ in range(runs):
            sim = DeterministicUAVSimulation(num_users=num_users)
            _, user_rates = sim.run()
            total_fairness += jains_fairness(user_rates)
        avg_fairness = total_fairness / runs
        fairness_scores.append(avg_fairness)


    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, fairness_scores, marker='o', linestyle='-', color='darkblue')
    plt.xlabel("Number of Users (UE Density)")
    plt.xticks(user_rates)
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness vs UE Density (Deterministic)")
    plt.grid(True)
    plt.xticks(user_counts)
    plt.tight_layout()
    plt.savefig("results/deterministic_fairness_vs_density.png")
    plt.show()

    # Save fairness values for later comparison
    np.save("results/deterministic_fairness_vs_density.npy", np.array(fairness_scores))

 
    min_rate_scores = []

    for num_users in user_counts:
        total_min_rate = 0
        runs = 1000
        for _ in range(runs):
            sim = DeterministicUAVSimulation(num_users=num_users)
            _, user_rates = sim.run()
            total_min_rate += np.min(user_rates)
        avg_min_rate = total_min_rate / runs
        min_rate_scores.append(avg_min_rate)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, min_rate_scores, marker='s', linestyle='-', color='darkred')
    plt.xlabel("Number of Users (UE Density)")
    plt.ylabel("Minimum Throughput per UE (Mbps)")
    plt.title("Min Rate vs UE Density (Deterministic)")
    plt.grid(True)
    plt.xticks(user_counts)
    plt.tight_layout()
    plt.savefig("results/deterministic_min_rate_vs_density.png")
    plt.show()

    # Save result
    np.save("results/deterministic_min_rate_vs_density.npy", np.array(min_rate_scores))
