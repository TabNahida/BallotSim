import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches

class VotingSimulation:
    def __init__(self,
                 num_voters=100,
                 num_candidates=3,
                 center=np.array([0, 0]),
                 spread_radius=10,
                 dimension=2):
        """
        Initialize a voting simulation.

        num_voters:       Number of voters
        num_candidates:   Number of candidates/proposals (at least 2)
        center:           The "center point" for generating voter and candidate positions (default [0, 0])
        spread_radius:    Maximum radius for distributing voters and candidates
        dimension:        Dimensionality (currently only 2D visualization is supported)
        """
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.center = np.array(center, dtype=float)
        self.spread_radius = spread_radius
        self.dimension = dimension

        # Assign a constant color to each candidate
        cmap_name = 'tab10' if self.num_candidates <= 10 else 'tab20'
        cmap = plt.cm.get_cmap(cmap_name, max(self.num_candidates, 10))
        self.cand_colors = [cmap(i) for i in range(self.num_candidates)]

        # Generate initial voter and candidate positions
        self.voters = self._generate_voters(self.center)
        self.candidates = self._generate_candidates(self.center)
        self.votes = None  # To store which candidate each voter voted for (indices 0..num_candidates-1)

    def _generate_voters(self, center):
        """
        Generate (num_voters - 1) random points inside a circle (angles uniform,
        radii ~ sqrt(uniform) * spread_radius), then compute a compensating point
        so that the centroid is exactly at center.
        """
        angles = np.random.uniform(0, 2 * np.pi, self.num_voters - 1)
        radii = np.sqrt(np.random.uniform(0, 1, self.num_voters - 1)) * self.spread_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        raw = np.column_stack([x, y])  # Shape (num_voters - 1, 2)

        # Compensating point = -sum of the above points, to ensure centroid at (0,0), then shift to center
        last = -np.sum(raw, axis=0)
        voters = np.vstack([raw, last]) + center
        return voters  # Shape (num_voters, 2)

    def _generate_candidates(self, center):
        """
        Generate num_candidates random candidate coordinates inside a circle
        (uniform angles, radii ~ sqrt(uniform) * spread_radius), then shift to center.
        """
        angles = np.random.uniform(0, 2 * np.pi, self.num_candidates)
        radii = np.sqrt(np.random.uniform(0, 1, self.num_candidates)) * self.spread_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y]) + center  # Shape (num_candidates, 2)

    def vote(self):
        """
        Perform one round of voting: each voter computes distances to all candidates
        and votes for the nearest one. Results are saved in self.votes as an array of length num_voters.
        """
        distances = np.linalg.norm(
            self.voters[:, np.newaxis, :] - self.candidates[np.newaxis, :, :],
            axis=2
        )  # Shape (num_voters, num_candidates)

        self.votes = np.argmin(distances, axis=1)

    def get_result(self):
        """
        Print formatted voting results ("C0: xx", "C1: yy", ...). If voting hasn't occurred yet, run vote() first.
        """
        if self.votes is None:
            self.vote()
        counts = Counter(self.votes)
        print("Voting results:")
        for i in range(self.num_candidates):
            print(f"  C{i}: {counts.get(i, 0)}")

    def plot(self):
        """
        Plot voters, candidates, and voter centroid:
          1. Compute voter centroid (voter_center).
          2. Compute each voter's distance to centroid, take 95th percentile R95, and plot only the main cluster.
          3. Color supporters by their chosen candidate.
          4. Winner candidate in gold, others in red; border color uses self.cand_colors.
          5. Mark the voter centroid with a distinct marker and label.
          6. Legend in the top-right: "Voters (main cluster)" uses the color of the winning candidate.
        If voting hasn't occurred yet, run vote() first.
        """
        if self.dimension != 2:
            print("Currently only 2D visualization is supported")
            return

        if self.votes is None:
            self.vote()

        # (A) Compute voter centroid
        voter_center = np.mean(self.voters, axis=0)

        # (B) Distances of each voter to the centroid
        diffs = self.voters - voter_center
        dists = np.linalg.norm(diffs, axis=1)

        # (C) 95th percentile as threshold R95
        R95 = np.percentile(dists, 95)

        # Small margin to avoid cutting off points exactly on the boundary
        margin = self.spread_radius * 0.05

        # (D) Mask for main cluster voters
        main_mask = dists <= R95

        # (E) Assign colors to each voter by their chosen candidate
        voter_colors = [self.cand_colors[v] for v in self.votes]

        # (F) Determine winner index
        counts = Counter(self.votes)
        winner = max(range(self.num_candidates), key=lambda i: counts.get(i, 0))

        plt.figure(figsize=(8, 8))

        # Plot only main-cluster voters; no label (use proxy in legend)
        plt.scatter(self.voters[main_mask, 0],
                    self.voters[main_mask, 1],
                    c=np.array(voter_colors)[main_mask],
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.3)

        # (G) Plot candidates: winner in gold, others in red; borders with self.cand_colors[idx]
        for idx, cand in enumerate(self.candidates):
            if idx == winner:
                facecolor = 'gold'
                edgecolor = self.cand_colors[idx]
                size = 200
            else:
                facecolor = 'red'
                edgecolor = self.cand_colors[idx]
                size = 150
            plt.scatter(cand[0], cand[1],
                        c=facecolor,
                        edgecolors=edgecolor,
                        marker='X',
                        s=size,
                        linewidths=2,
                        zorder=5)
            # Label "C0, C1, ..." with border color
            plt.text(cand[0], cand[1], f'C{idx}',
                     fontsize=12, ha='center', va='center',
                     color='white',
                     bbox=dict(facecolor=edgecolor,
                               edgecolor='black',
                               boxstyle='circle,pad=0.3'),
                     zorder=6)

        # (H) Mark voter centroid
        plt.scatter(voter_center[0], voter_center[1],
                    c='black', marker='*', s=200, zorder=7)
        plt.text(voter_center[0], voter_center[1], '  Voter Center',
                 fontsize=11, ha='left', va='bottom', color='black',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                 zorder=8)

        # (I) Set axis limits around voter_center ± (R95 + margin)
        lim = R95 + margin
        plt.xlim(voter_center[0] - lim, voter_center[0] + lim)
        plt.ylim(voter_center[1] - lim, voter_center[1] + lim)
        plt.axis('equal')
        plt.grid(True)

        # (J) Legend: "Voters (main cluster)" in winner's color
        proxy_voter = mpatches.Patch(color=self.cand_colors[winner],
                                     label='Voters (main cluster)')
        plt.legend(handles=[proxy_voter], loc='upper right')

        plt.title("Voting Simulation")
        plt.show()

    def execute(self, change_func=None, **kwargs):
        """
        Move voters according to a specified strategy.
        By default (no change_func), perform a random walk.
        To apply a custom logic, pass change_func(simulation, **kwargs).
        """
        if change_func is None:
            # Default: random walk for voters
            random_walk(self, **kwargs)
        else:
            change_func(self, **kwargs)
        # Note: vote() is NOT automatically called after movement. Call sim.vote() if you want a new round.

    def propose(self, change_func=None, **kwargs):
        """
        Move or update candidates according to a specified strategy.
        By default (no change_func), perform a random walk for candidates.
        To use custom logic, pass change_func(simulation, **kwargs) which should modify sim.candidates.
        After proposing, previous vote results are reset; call sim.vote() again if needed.
        """
        if change_func is None:
            # Default: random walk for candidates
            candidate_random_walk(self, **kwargs)
        else:
            change_func(self, **kwargs)

        # Reset previous voting results; call vote() for the next round.
        self.votes = None


# ----------------- Candidate random walk -----------------
def candidate_random_walk(sim: VotingSimulation, step_size=1.0):
    """
    Add independent Gaussian noise to each candidate's coordinates (random walk).
    step_size: standard deviation of the noise.
    """
    sim.candidates += np.random.normal(loc=0.0,
                                       scale=step_size,
                                       size=sim.candidates.shape)


# ----------------- Move candidates toward voter centroid -----------------
def evolve_towards_voter_center(sim: VotingSimulation, fraction=0.2):
    """
    Move each candidate a fraction of the way toward the current voter centroid.
    fraction=0.2 means moving 20% of the distance each call.
    """
    voter_center = np.mean(sim.voters, axis=0)
    directions = voter_center - sim.candidates  # Shape (num_candidates, 2)
    sim.candidates = sim.candidates + fraction * directions


# ----------------- Move candidates away from voter centroid (inverse) -----------------
def evolve_away_from_voter_center(sim: VotingSimulation, fraction=0.2):
    """
    Move each candidate a fraction of the way away from the current voter centroid.
    fraction=0.2 means moving 20% of the distance each call.
    """
    voter_center = np.mean(sim.voters, axis=0)
    directions = sim.candidates - voter_center  # Reverse direction
    sim.candidates = sim.candidates + fraction * directions


# ----------------- Add Gaussian noise to candidates -----------------
def evolve_add_noise(sim: VotingSimulation, noise_std=1.0):
    """
    Add independent Gaussian noise to each candidate's position.
    noise_std: standard deviation of the noise.
    """
    noise = np.random.normal(loc=0.0, scale=noise_std,
                             size=sim.candidates.shape)
    sim.candidates = sim.candidates + noise


# ----------------- Random walk for voters -----------------
def random_walk(sim: VotingSimulation, step_size=1.0):
    """
    Move all voters by adding Gaussian noise to their positions.
    step_size: standard deviation of the noise.
    """
    sim.voters += np.random.normal(loc=0.0, scale=step_size, size=sim.voters.shape)


# ----------------- Move voters toward the winner -----------------
def move_towards_winner(sim: VotingSimulation, fraction=0.1):
    """
    Ensure voting has occurred at least once. Compute the winner (most votes),
    then move each voter a fraction of the way toward that winner.
    fraction=0.1 means moving 10% of the distance each call.
    """
    if sim.votes is None:
        sim.vote()
    counts = Counter(sim.votes)
    winner = max(range(sim.num_candidates), key=lambda i: counts.get(i, 0))
    winner_pos = sim.candidates[winner]
    directions = winner_pos - sim.voters
    sim.voters = sim.voters + fraction * directions


# ----------------- Probabilistic move toward or away from the winner -----------------
def probabilistic_move_towards_winner(sim: VotingSimulation, fraction=0.1):
    """
    Based on the most recent vote results, identify the winning candidate.
    Compute A = mean distance of all voters to the winner (using the winner's current position).
    For each voter i, let B_i = distance of voter i to the winner.
    The probability p_i = B_i / (A + B_i) that voter i moves TOWARD the winner by distance (fraction * A).
    Otherwise, the voter moves AWAY from the winner by the same distance (fraction * A).
    If a voter is exactly at the winner's position (B_i == 0), no movement occurs for that voter.
    """
    if sim.votes is None:
        sim.vote()

    counts = Counter(sim.votes)
    winner = max(range(sim.num_candidates), key=lambda i: counts.get(i, 0))
    winner_pos = sim.candidates[winner]  # Use current winner position

    # Compute distances B_i for each voter
    diffs = winner_pos - sim.voters  # Shape (num_voters, 2)
    B = np.linalg.norm(diffs, axis=1)  # Shape (num_voters,)

    # Compute A = average distance
    A = np.mean(B)

    if A == 0:
        # All voters are exactly at the winner position; no movement
        return

    # Distance each voter should move: fraction * A
    move_distance = fraction * A

    # For voters with B_i > 0, compute unit direction vectors
    nonzero_mask = B > 0
    unit_dirs = np.zeros_like(diffs)
    unit_dirs[nonzero_mask] = diffs[nonzero_mask] / B[nonzero_mask, np.newaxis]

    # Compute probabilities p_i = B_i / (A + B_i)
    p = np.zeros_like(B)
    p[nonzero_mask] = B[nonzero_mask] / (A + B[nonzero_mask])

    # Sample uniform random numbers to decide movement direction
    rand_vals = np.random.uniform(0, 1, size=B.shape)

    # Move each voter accordingly
    for i in range(sim.num_voters):
        if B[i] == 0:
            # Voter exactly at winner position: no movement
            continue
        if rand_vals[i] < p[i]:
            # Move toward the winner by move_distance
            sim.voters[i] += unit_dirs[i] * move_distance
        else:
            # Move away from the winner by move_distance
            sim.voters[i] -= unit_dirs[i] * move_distance


# ----------------- Cluster attraction: move outermost voters toward centroid -----------------
def cluster_attraction(sim: VotingSimulation, cluster_pct=0.2):
    """
    Compute the centroid of all voters. Identify the top cluster_pct fraction of voters
    farthest from the centroid, and move them a small step (5% of their distance) toward the centroid.
    Remaining voters stay in place.
    cluster_pct: fraction (e.g., 0.2 means the farthest 20% move).
    """
    center = np.mean(sim.voters, axis=0)
    dists = np.linalg.norm(sim.voters - center, axis=1)
    threshold = np.percentile(dists, 100 * (1 - cluster_pct))
    mask = dists >= threshold
    directions = center - sim.voters[mask]
    sim.voters[mask] = sim.voters[mask] + 0.05 * directions
