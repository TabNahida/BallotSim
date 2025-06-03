import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

        # ------------- 新增：为每个候选人预先分配一个“恒定颜色” ---------------
        # 生成一个 colormap，确保颜色数量 >= num_candidates
        cmap_name = 'tab10' if self.num_candidates <= 10 else 'tab20'
        cmap = plt.cm.get_cmap(cmap_name, max(self.num_candidates, 10))
        # 存成列表：比如 cand_colors[i] 就是编号 i 的候选人颜色
        self.cand_colors = [cmap(i) for i in range(self.num_candidates)]
        # ---------------------------------------------------------------

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
        # 生成 (num_voters, num_candidates) 的距离矩阵
        distances = np.linalg.norm(
            self.voters[:, np.newaxis, :] - self.candidates[np.newaxis, :, :],
            axis=2
        )  # 结果形状 (num_voters, num_candidates)

        # argmin 返回每行最小值的索引，也就是每个选民投给哪个候选人（索引）
        self.votes = np.argmin(distances, axis=1)

    def get_result(self):
        """
        打印格式化后的投票结果（每一行 "C0: xx", "C1: yy", ...）。
        如果还没投票，就先执行一次 vote()。
        """
        if self.votes is None:
            self.vote()
        counts = Counter(self.votes)  # 统计每个候选人得票数
        print("投票结果：")
        for i in range(self.num_candidates):
            print(f"  C{i}: {counts.get(i, 0)}")

    def plot(self):
        """
        将主要选民群体、候选人、选民中心绘制出来：
          1. 先算出选民重心 voter_center；
          2. 计算每个选民到 voter_center 的距离，取 95% 分位数 R95，只画主要簇内的选民；
          3. 支持者按不同候选人分配不同颜色；
          4. 赢家候选人用金色 (gold)，其它候选人用红色 (red)；需沿用恒定的 self.cand_colors；
          5. 在图上标出选民质心，并加一个明显的标注。
        如果还没投票，就先执行一次 vote()。
        """
        if self.dimension != 2:
            print("目前仅支持二维可视化")
            return

        if self.votes is None:
            self.vote()

        # （A）计算选民重心
        voter_center = np.mean(self.voters, axis=0)  # 形如 [cx, cy]

        # （B）算出每个选民到重心的距离 d_i
        diffs = self.voters - voter_center  # 形状 (num_voters, 2)
        dists = np.linalg.norm(diffs, axis=1)  # 形状 (num_voters, )

        # （C）取 95% 分位数作为阈值 R95
        R95 = np.percentile(dists, 95)

        # 留一点小边距，防止点正好在边界处被切掉
        margin = self.spread_radius * 0.05  # 半径的 5%

        # （D）确定要绘制的“主要选民”索引
        main_mask = dists <= R95  # 布尔数组，True 表示此选民在主要簇内

        plt.figure(figsize=(8, 8))

        # （E）为每个选民分配一种不重复的颜色（按 self.votes 和 self.cand_colors）
        voter_colors = [self.cand_colors[v] for v in self.votes]

        # 只画 main_mask == True 的选民
        plt.scatter(self.voters[main_mask, 0],
                    self.voters[main_mask, 1],
                    c=np.array(voter_colors)[main_mask],
                    label='Voters (main cluster)',
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.3)

        # （F）统计得票情况并找出“赢家”索引
        counts = Counter(self.votes)
        winner = max(range(self.num_candidates), key=lambda i: counts.get(i, 0))

        # （G）绘制候选人：赢家金色、其它红色，
        #     但为了“颜色一致”，我们在边框上仍然用 self.cand_colors[idx] 标示各自编号
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

        plt.title("Voting Simulation")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.show()

    def execute(self, change_func=None, **kwargs):
        """
        让选民按照一定策略“移动”一次。默认情况下（change_func=None），选民位置不变。
        如果要改变选民位置，可以传入一个函数 change_func(simulation, **kwargs)，
        它接收当前的 VotingSimulation 对象和额外的参数，然后在函数内部修改 sim.voters。
        kwargs 会原样传给 change_func。
        """
        if change_func is not None:
            change_func(self, **kwargs)
        # 如果 change_func=None，则什么都不做（选民保持原位）

    def propose(self,
                new_num_candidates=None,
                mode='random',
                evolve_func=None,
                **kwargs):
        """
        重新生成下一轮候选人，并可选地调整候选人数量、指定“演化”函数。

        参数：
          - new_num_candidates: 如果想改变候选人数量，就传入一个整数；否则保持原来的 self.num_candidates。
          - mode: 'random' 或 'evolve'，分别表示：
                * 'random'：直接在当前选民质心处随机重新生成 new_num_candidates 个候选人；
                * 'evolve'：基于当前候选人 self.candidates，按某种规则生成“演化后”的新位置；
                            需要用户自己提供 evolve_func，否则会报错。
          - evolve_func: 如果 mode='evolve'，则此参数必须是一个函数，函数签名为：
                def my_evolve(sim: VotingSimulation, **kwargs) -> np.ndarray
            它接收当前的 sim 对象，并返回一个形状为 (new_num_candidates, 2) 的数组，表示新的候选人位置。
            例如，可以把它写成“给每个候选人加一点随机扰动”或者“向选民中心靠拢一小步”之类的逻辑。
          - kwargs: 额外参数，会原样传给 evolve_func(sim, **kwargs)。

        使用示例：
            sim.propose(new_num_candidates=4, mode='random')
            # 或者
            sim.propose(mode='evolve', evolve_func=my_evolve, step_size=0.5)
        """
        # 1. 先处理候选人数量的变化
        if new_num_candidates is not None:
            # 如果改变了数量，需要重新生成 self.cand_colors 列表
            old_num = self.num_candidates
            self.num_candidates = new_num_candidates
            # 重新创建一套颜色，确保每个编号都有自己的颜色
            cmap_name = 'tab10' if self.num_candidates <= 10 else 'tab20'
            cmap = plt.cm.get_cmap(cmap_name, max(self.num_candidates, 10))
            self.cand_colors = [cmap(i) for i in range(self.num_candidates)]
        else:
            # 数量不变，保持原来的 self.num_candidates 和 self.cand_colors
            pass

        # 2. 根据 mode 决定如何生成新候选人
        # 先算出当前选民质心
        voter_center = np.mean(self.voters, axis=0)

        if mode == 'random':
            # 直接随机：在 voter_center 周围的圆圈内重采样
            self.candidates = self._generate_candidates(voter_center)

        elif mode == 'evolve':
            # 演化模式：必须提供 evolve_func 来产生新位置
            if evolve_func is None:
                raise ValueError("mode='evolve' 时必须传入 evolve_func 参数")
            new_positions = evolve_func(self, **kwargs)
            # 检查返回值尺寸是否匹配
            new_positions = np.array(new_positions, dtype=float)
            if new_positions.shape != (self.num_candidates, 2):
                raise ValueError(f"evolve_func 返回数组尺寸应为 ({self.num_candidates}, 2)，"
                                 f"但得到 {new_positions.shape}")
            self.candidates = new_positions

        else:
            raise ValueError("mode 必须是 'random' 或 'evolve'")

        # 3. 重置之前的投票结果，下一轮再 call vote()
        self.votes = None

def evolve_towards_voter_center(sim: VotingSimulation, fraction=0.2):
    """
    让每个候选人都朝着当前选民质心移动一小段距离，
    fraction=0.2 表示每次跨过去 20% 的距离。
    返回形状为 (sim.num_candidates, 2) 的新位置数组。
    """
    voter_center = np.mean(sim.voters, axis=0)
    directions = voter_center - sim.candidates  # 大小 (num_candidates, 2)
    new_pos = sim.candidates + fraction * directions
    return new_pos


def evolve_add_noise(sim: VotingSimulation, noise_std=1.0):
    """
    给每个候选人位置都加一个独立的高斯噪声，标准差为 noise_std。
    返回一个 (num_candidates, 2) 大小的数组。
    """
    noise = np.random.normal(loc=0.0, scale=noise_std,
                             size=sim.candidates.shape)
    return sim.candidates + noise

def random_walk(sim, step_size=1.0):
    """
    sim: VotingSimulation 对象
    step_size: 标准差，表示随机步长的尺度
    """
    sim.voters += np.random.normal(loc=0.0, scale=step_size, size=sim.voters.shape)

def move_towards_winner(sim, fraction=0.1):
    """
    先确保已有一次投票结果，再计算赢家（得票数最多的候选人），
    然后把每个选民沿着自己到赢家候选人的方向前进 fraction（0~1）距离。
    
    fraction=0.1 意味着每次只走 10% 的距离。
    """
    if sim.votes is None:
        sim.vote()
    # 计算赢家索引
    counts = Counter(sim.votes)
    winner = max(range(sim.num_candidates), key=lambda i: counts.get(i, 0))
    winner_pos = sim.candidates[winner]  # 赢家的坐标
    
    # 对每个选民，让其向赢家移动
    directions = winner_pos - sim.voters  # 形状 (num_voters, 2)
    sim.voters = sim.voters + fraction * directions

def cluster_attraction(sim, cluster_pct=0.2):
    """
    先计算所有选民的质心，然后把离质心最远的 cluster_pct 比例的选民，
    往质心方向移动一小步；其余选民不变。模拟“中心势力”吸引边缘选民。
    
    cluster_pct: 比如 0.2 表示最远 20% 的选民会向中心靠拢。
    """
    # 1. 计算选民质心
    center = np.mean(sim.voters, axis=0)
    # 2. 计算每个选民到质心的距离
    dists = np.linalg.norm(sim.voters - center, axis=1)
    # 3. 找到最远 cluster_pct 的阈值
    threshold = np.percentile(dists, 100 * (1 - cluster_pct))
    # 4. 对最远的选民，向质心移动一个固定步长（例如 5% 距离）
    mask = dists >= threshold
    directions = center - sim.voters[mask]
    sim.voters[mask] = sim.voters[mask] + 0.05 * directions
    # 剩余选民保持不变
