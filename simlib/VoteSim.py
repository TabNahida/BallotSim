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
        num_voters:       选民数量
        num_candidates:   候选人/提案数量（至少 2）
        center:           生成选民、候选人的“中心点”（默认为 [0,0]）
        spread_radius:    选民和候选人分布时的最大半径范围
        dimension:        维度（目前只支持二维可视化）
        """
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.center = np.array(center, dtype=float)
        self.spread_radius = spread_radius
        self.dimension = dimension

        # 为每个候选人预先分配一个“恒定颜色”
        cmap_name = 'tab10' if self.num_candidates <= 10 else 'tab20'
        cmap = plt.cm.get_cmap(cmap_name, max(self.num_candidates, 10))
        self.cand_colors = [cmap(i) for i in range(self.num_candidates)]

        # 初始化时生成选民和候选人位置
        self.voters = self._generate_voters(self.center)
        self.candidates = self._generate_candidates(self.center)
        self.votes = None  # 用于储存每个选民投票给哪个候选人（索引列表）

    def _generate_voters(self, center):
        """
        在一个圆形范围内生成 (num_voters - 1) 个随机点（角度均匀，半径 ~ sqrt(uniform)*spread_radius），
        然后第 N 个点作为补偿，使得所有点质心在 center。
        """
        angles = np.random.uniform(0, 2 * np.pi, self.num_voters - 1)
        radii = np.sqrt(np.random.uniform(0, 1, self.num_voters - 1)) * self.spread_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        raw = np.column_stack([x, y])  # 形状 (N-1, 2)

        # 补偿点 = －(前面所有点向量和)，保证质心在 (0,0) 然后整体平移到 center
        last = -np.sum(raw, axis=0)
        voters = np.vstack([raw, last]) + center
        return voters  # 形状 (N, 2)

    def _generate_candidates(self, center):
        """
        在同样的圆形范围内，均匀随机生成 num_candidates 个候选人坐标，最后平移到 center。
        """
        angles = np.random.uniform(0, 2 * np.pi, self.num_candidates)
        radii = np.sqrt(np.random.uniform(0, 1, self.num_candidates)) * self.spread_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y]) + center  # 形状 (num_candidates, 2)

    def vote(self):
        """
        执行一次投票：每个选民计算到各候选人的距离，并将票投给最近的那个候选人。
        结果保存在 self.votes，这是一个长度为 num_voters 的整数数组（0..num_candidates-1）。
        """
        distances = np.linalg.norm(
            self.voters[:, np.newaxis, :] - self.candidates[np.newaxis, :, :],
            axis=2
        )  # 结果形状 (num_voters, num_candidates)

        self.votes = np.argmin(distances, axis=1)

    def get_result(self):
        """
        打印格式化后的投票结果（每一行 "C0: xx", "C1: yy", ...）。
        如果还没投票，就先执行一次 vote()。
        """
        if self.votes is None:
            self.vote()
        counts = Counter(self.votes)
        print("投票结果：")
        for i in range(self.num_candidates):
            print(f"  C{i}: {counts.get(i, 0)}")

    def plot(self):
        """
        将主要选民群体、候选人、选民中心绘制出来：
          1. 先算出选民重心 voter_center；
          2. 计算每个选民到 voter_center 的距离，取 95% 分位数 R95，只画主要簇内的选民；
          3. 支持者按不同候选人分配不同颜色；
          4. 赢家候选人用金色 (gold)，其它候选人用红色 (red)；但在标注边框和编号时仍用 self.cand_colors；
          5. 在图上标出选民质心，并加一个明显的标注；
          6. 右上角图例中，“Voters (main cluster)” 的颜色与得票最多的候选人保持一致。
        如果还没投票，就先执行一次 vote()。
        """
        if self.dimension != 2:
            print("目前仅支持二维可视化")
            return

        if self.votes is None:
            self.vote()

        # （A）计算选民重心
        voter_center = np.mean(self.voters, axis=0)

        # （B）算出每个选民到重心的距离 d_i
        diffs = self.voters - voter_center
        dists = np.linalg.norm(diffs, axis=1)

        # （C）取 95% 分位数作为阈值 R95
        R95 = np.percentile(dists, 95)

        # 留一点小边距，防止点正好在边界处被切掉
        margin = self.spread_radius * 0.05

        # （D）确定要绘制的“主要选民”索引
        main_mask = dists <= R95

        # （E）为每个选民分配一种颜色（按 self.votes 和 self.cand_colors）
        voter_colors = [self.cand_colors[v] for v in self.votes]

        # （F）统计得票情况并找出“赢家”索引
        counts = Counter(self.votes)
        winner = max(range(self.num_candidates), key=lambda i: counts.get(i, 0))

        plt.figure(figsize=(8, 8))

        # 只画 main_mask==True 的选民，但不带 label，用一个 proxy 在图例里显示
        plt.scatter(self.voters[main_mask, 0],
                    self.voters[main_mask, 1],
                    c=np.array(voter_colors)[main_mask],
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.3)

        # （G）绘制候选人：赢家金色、其它红色，边框用 self.cand_colors[idx]
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
            # 在点上写上 “C0, C1, ...” 并用边框颜色突出编号含义
            plt.text(cand[0], cand[1], f'C{idx}',
                     fontsize=12, ha='center', va='center',
                     color='white',
                     bbox=dict(facecolor=edgecolor,
                               edgecolor='black',
                               boxstyle='circle,pad=0.3'),
                     zorder=6)

        # （H）在图上标出选民重心
        plt.scatter(voter_center[0], voter_center[1],
                    c='black', marker='*', s=200, zorder=7)
        plt.text(voter_center[0], voter_center[1], '  Voter Center',
                 fontsize=11, ha='left', va='bottom', color='black',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                 zorder=8)

        # （I）设置坐标轴范围：以 voter_center 为中心，±(R95 + margin)
        lim = R95 + margin
        plt.xlim(voter_center[0] - lim, voter_center[0] + lim)
        plt.ylim(voter_center[1] - lim, voter_center[1] + lim)
        plt.axis('equal')
        plt.grid(True)

        # （J）制作图例：Voters (main cluster) 用 winner 的颜色，其它可根据需要自行添加
        # 这里只演示将 Voters (main cluster) 用得票最多的候选人颜色：
        proxy_voter = mpatches.Patch(color=self.cand_colors[winner],
                                     label='Voters (main cluster)')
        plt.legend(handles=[proxy_voter], loc='upper right')

        plt.title("Voting Simulation")
        plt.show()

    # ----------------- 修改：execute 默认使用 random_walk -----------------
    def execute(self, change_func=None, **kwargs):
        """
        让选民按照一定策略“移动”一次。
        默认情况下（不传入 change_func），会执行 random_walk。
        如果要按照其它逻辑移动选民，则传入 change_func(simulation, **kwargs)。
        """
        if change_func is None:
            # 默认使用随机游走
            random_walk(self, **kwargs)
        else:
            change_func(self, **kwargs)
        # 注意：不再自动调用 vote()，如果移动后需要新一轮投票，请手动再调用 vote()。

    # ----------------- 修改：propose 去掉 mode，只考虑一个函数 -----------------
    def propose(self, change_func=None, **kwargs):
        """
        让候选人按照一定策略“移动”或“更新”一次。默认情况下（不传入 change_func），
        会对候选人执行随机游走（candidate_random_walk）。

        如果要按自定义逻辑生成/演化候选人位置，则传入 change_func(simulation, **kwargs)，
        该函数应修改 sim.candidates。

        参数：
          - change_func: 函数签名应为 change_func(sim: VotingSimulation, **kwargs)，
                         接收当前的 sim 对象，并在内部修改 sim.candidates。
                         如果为 None，则默认使用 candidate_random_walk。
          - kwargs: 额外参数，会原样传给 change_func(sim, **kwargs)。
        """
        if change_func is None:
            # 默认对候选人执行随机游走
            candidate_random_walk(self, **kwargs)
        else:
            change_func(self, **kwargs)

        # 重置之前的投票结果，下一轮再调用 vote()
        self.votes = None


# ----------------- 新增：对候选人做随机游走的函数 -----------------
def candidate_random_walk(sim: VotingSimulation, step_size=1.0):
    """
    对每个候选人的坐标都加一个独立的高斯噪声（随机游走）。
    step_size: 随机步长的尺度，相当于标准差。
    """
    sim.candidates += np.random.normal(loc=0.0,
                                       scale=step_size,
                                       size=sim.candidates.shape)


# 以下保留你之前定义的演化函数等（如果你还需要的话）
def evolve_towards_voter_center(sim: VotingSimulation, fraction=0.2):
    """
    让每个候选人都朝着当前选民质心移动一小段距离，
    fraction=0.2 表示每次跨过去 20% 的距离。
    返回形状为 (sim.num_candidates, 2) 的新位置数组。
    """
    voter_center = np.mean(sim.voters, axis=0)
    directions = voter_center - sim.candidates  # 大小 (num_candidates, 2)
    new_pos = sim.candidates + fraction * directions
    sim.candidates = new_pos

def evolve_add_noise(sim: VotingSimulation, noise_std=1.0):
    """
    给每个候选人位置都加一个独立的高斯噪声，标准差为 noise_std。
    返回一个 (num_candidates, 2) 大小的数组。
    """
    noise = np.random.normal(loc=0.0, scale=noise_std,
                             size=sim.candidates.shape)
    sim.candidates = sim.candidates + noise

def random_walk(sim: VotingSimulation, step_size=1.0):
    """
    sim: VotingSimulation 对象
    step_size: 标准差，表示随机步长的尺度
    对所有选民执行随机游走。
    """
    sim.voters += np.random.normal(loc=0.0, scale=step_size, size=sim.voters.shape)

def move_towards_winner(sim: VotingSimulation, fraction=0.1):
    """
    先确保已有一次投票结果，再计算赢家（得票数最多的候选人），
    然后把每个选民沿着自己到赢家候选人的方向前进 fraction（0~1）距离。
    fraction=0.1 意味着每次只走 10% 的距离。
    """
    if sim.votes is None:
        sim.vote()
    counts = Counter(sim.votes)
    winner = max(range(sim.num_candidates), key=lambda i: counts.get(i, 0))
    winner_pos = sim.candidates[winner]
    directions = winner_pos - sim.voters
    sim.voters = sim.voters + fraction * directions

def cluster_attraction(sim: VotingSimulation, cluster_pct=0.2):
    """
    先计算所有选民的质心，然后把离质心最远的 cluster_pct 比例的选民，
    往质心方向移动一小步；其余选民不变。模拟“中心势力”吸引边缘选民。
    cluster_pct: 比如 0.2 表示最远 20% 的选民会向中心靠拢。
    """
    center = np.mean(sim.voters, axis=0)
    dists = np.linalg.norm(sim.voters - center, axis=1)
    threshold = np.percentile(dists, 100 * (1 - cluster_pct))
    mask = dists >= threshold
    directions = center - sim.voters[mask]
    sim.voters[mask] = sim.voters[mask] + 0.05 * directions

