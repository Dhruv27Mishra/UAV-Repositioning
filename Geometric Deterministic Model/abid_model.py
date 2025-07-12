import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import argparse
import os

class AbidUAVSimulation:
    def __init__(self, num_uavs=3, num_users=20, grid_size=(10, 10), uav_altitude=12.0, tx_power=1.0, noise_power=1e-3, path_loss_exp=2.5, bandwidth=1.0):
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.grid_size = grid_size
        self.uav_altitude = uav_altitude
        self.tx_power = tx_power
        self.noise_power = noise_power
        self.path_loss_exp = path_loss_exp
        self.bandwidth = bandwidth
        
        # Energy model parameters (same as MARLEnv)
        self.uav_mass = 1.5  # kg
        self.uav_area = 0.25  # m^2
        self.uav_s = 0.05  # rotor solidity
        self.uav_P0 = 23.0  # blade profile power (W)
        self.uav_Pi = 88.6  # induced power in hover (W)
        self.uav_Utip = 120.0  # rotor tip speed (m/s)
        self.uav_v0 = 4.03  # mean rotor induced velocity in hover (m/s)
        self.uav_d0 = 0.012  # fuselage drag ratio
        self.uav_rho = 1.225  # air density (kg/m^3)

        self.user_positions = self._initialize_user_positions()
        self.uav_positions = self._initialize_uav_positions()
        self.user_rates = np.zeros(self.num_users)
        self.association = np.zeros(self.num_users, dtype=int)

    def _initialize_user_positions(self):
        """Initialize user positions using Poisson point process."""
        # For Poisson point process, we can use uniform random but with proper density
        # The number of users already represents the Poisson intensity
        positions = np.random.rand(self.num_users, 3)
        positions[:, 0] *= self.grid_size[0]
        positions[:, 1] *= self.grid_size[1]
        positions[:, 2] = 0
        return positions

    def _initialize_uav_positions(self):
        # Place UAVs at the center of the minimum enclosing ball of all users
        xy = self.user_positions[:, :2]
        center, radius = self._min_enclosing_ball(xy)
        z = self.uav_altitude
        # Spread UAVs evenly around the center (for multiple UAVs)
        angles = np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
        uav_positions = np.array([
            [center[0] + radius * 0.2 * np.cos(a), center[1] + radius * 0.2 * np.sin(a), z]
            for a in angles
        ])
        return uav_positions

    def _min_enclosing_ball(self, points):
        # Welzl's algorithm for 2D minimum enclosing ball
        from scipy.spatial import distance_matrix
        center = np.mean(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        radius = np.max(dists)
        return center, radius

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

    def _compute_energy(self):
        """Compute energy consumption using the same model as MARLEnv."""
        # For Abid model, assume UAVs are stationary (velocity = 0)
        velocities = np.zeros(self.num_uavs)
        
        def uav_power(V):
            P0 = self.uav_P0
            Pi = self.uav_Pi
            Utip = self.uav_Utip
            v0 = self.uav_v0
            d0 = self.uav_d0
            s = self.uav_s
            A = self.uav_area
            rho = self.uav_rho
            # Profile power
            P_profile = P0 * (1 + 3 * (V**2) / (Utip**2))
            # Induced power
            P_induced = Pi * (np.sqrt(1 + (V**4) / (4 * v0**4)) - (V**2) / (2 * v0**2))**0.5
            # Parasite power
            P_parasite = 0.5 * d0 * rho * s * A * (V**3)
            return P_profile + P_induced + P_parasite
        
        uav_powers = np.array([uav_power(V) for V in velocities])
        total_power = np.sum(uav_powers)
        # Assume dt=1 per step, so energy = power
        total_energy = total_power
        return total_energy

    def run(self):
        # Place UAVs using minimum enclosing ball of all eMBB users
        xy = self.user_positions[:, :2]
        center, radius = self._min_enclosing_ball(xy)
        z = self.uav_altitude
        angles = np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
        self.uav_positions = np.array([
            [center[0] + radius * 0.2 * np.cos(a), center[1] + radius * 0.2 * np.sin(a), z]
            for a in angles
        ])
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
        
        # Compute energy
        total_energy = self._compute_energy()
        
        return total_throughput, self.user_rates, total_energy

    def run_multiple_episodes(self, num_episodes=1000):
        """Run multiple episodes and return arrays of results."""
        throughputs = []
        energies = []
        all_user_rates = []  # Store user rates for each episode
        
        for episode in range(num_episodes):
            # Reinitialize user positions for each episode
            self.user_positions = self._initialize_user_positions()
            
            total_throughput, user_rates, total_energy = self.run()
            
            throughputs.append(total_throughput)
            energies.append(total_energy)
            all_user_rates.append(user_rates.copy())  # Store user rates for this episode
            
            # Per-episode printout
            print(f"Episode {episode+1} completed")
            print(f"Throughput: {total_throughput:.2f}")
            print(f"Energy: {total_energy:.2f}")
        
        return np.array(throughputs), np.array(energies), np.array(all_user_rates)

    def visualize(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.user_positions[:, 0], self.user_positions[:, 1], c='green', label='eMBB Users')
        plt.scatter(self.uav_positions[:, 0], self.uav_positions[:, 1], c='red', s=100, label='UAVs (Abid)')
        for j in range(self.num_users):
            user = self.user_positions[j, :2]
            uav = self.uav_positions[self.association[j], :2]
            plt.plot([user[0], uav[0]], [user[1], uav[1]], 'gray', alpha=0.4)
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Abid Model UAV Placement and User Associations (eMBB only)')
        plt.legend()
        plt.grid(True)
        plt.show()
def jains_fairness(user_rates):
    user_rates = np.array(user_rates)
    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)
    return numerator / denominator if denominator != 0 else 0

def run_fairness_vs_density():
    user_counts = [1, 2, 3, 4, 5]
    fairness_scores = []
    runs = 1000

    for num_users in user_counts:
        total_fairness = 0
        for _ in range(runs):
            sim = AbidUAVSimulation(num_uavs=5, num_users=num_users)
            _, user_rates, _ = sim.run()
            total_fairness += jains_fairness(user_rates)
        avg_fairness = total_fairness / runs
        fairness_scores.append(avg_fairness)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, fairness_scores, marker='o', linestyle='-', color='green')
    plt.xlabel("Number of Users (UE Density)")
    plt.xticks(user_counts)
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness vs UE Density (Abid Model)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/abid_fairness_vs_density.png")
    plt.show()

    np.save("results/abid_fairness_vs_density.npy", np.array(fairness_scores))
def run_simulation(num_episodes=500, num_uavs=7, num_users=49):
    """Run Abid simulation for multiple episodes and save results."""
    sim = AbidUAVSimulation(num_uavs=num_uavs, num_users=num_users)
    throughputs, energies, all_user_rates = sim.run_multiple_episodes(num_episodes)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    np.save("results/abid_throughputs.npy", throughputs)
    np.save("results/abid_energies.npy", energies)
    np.save("results/abid_user_rates.npy", all_user_rates)
    
    # Create dummy rewards array (Abid model doesn't have rewards)
    rewards = np.zeros_like(throughputs)
    np.save("results/abid_rewards.npy", rewards)
    
    # Calculate statistics
    total_throughput = np.sum(throughputs)
    total_energy = np.sum(energies)
    total_collisions = 0  # Abid model doesn't have collisions
    
    # Print final statistics
    print("\nSimulation completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average collisions per episode: {total_collisions/num_episodes:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")

def run_min_rate_vs_density():
    user_counts = [1, 2, 3, 4, 5]
    min_rates = []
    runs = 1000

    for num_users in user_counts:
        total_min = 0
        for _ in range(runs):
            sim = AbidUAVSimulation(num_uavs=5, num_users=num_users)
            _, user_rates, _ = sim.run()
            total_min += np.min(user_rates)
        avg_min = total_min / runs
        min_rates.append(avg_min)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, min_rates, marker='^', color='darkgreen')
    plt.xlabel("Number of Users (UE Density)")
    plt.xticks(user_counts)
    plt.ylabel("Minimum Throughput per UE (Mbps)")
    plt.title("Min Rate vs UE Density (Abid Model)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/abid_min_rate_vs_density.png")
    plt.show()

    np.save("results/abid_min_rate_vs_density.npy", np.array(min_rates))   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Abid Deterministic Model for UAV positioning')
    parser.add_argument('--num_uavs', type=int, default=7, help='Number of UAVs')
    parser.add_argument('--num_users', type=int, default=49, help='Number of users')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of simulation episodes')
    
    args = parser.parse_args()
    run_fairness_vs_density()
    run_simulation(num_episodes=args.num_episodes, num_uavs=args.num_uavs, num_users=args.num_users) 
    run_min_rate_vs_density()