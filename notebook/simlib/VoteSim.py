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

        # 初始化时生成选民和候选人位置
        self.voters = self._generate_voters(self.center)
        self.candidates = self._generate_candidates(self.center)
        self.votes = None  # 用于储存每个选民投票给哪个候选人（索引列表）

    def _generate_voters(self, center):
        """
        在一个圆形范围内生成 (num_voters - 1) 个随机点（角度均匀，半径 ~ sqrt(uniform)*spread_radius），
        然后第 N 个点作为补偿，使得所有点质心在 (0,0)，最后整体平移到 center。
        """
        angles = np.random.uniform(0, 2 * np.pi, self.num_voters - 1)
        radii = np.sqrt(np.random.uniform(0, 1, self.num_voters - 1)) * self.spread_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        raw = np.column_stack([x, y])  # 形状 (N-1, 2)

        # 补偿点 = －(前面所有点向量和)，保证质心在 (0,0)
        last = -np.sum(raw, axis=0)
        voters = np.vstack([raw, last]) + center  # 平移到 center
        return voters  # 形状 (N, 2)

    def _generate_candidates(self, center):
        """
        在同样的圆形范围内，均匀随机生成 num_candidates 个候选人坐标，
        最后平移到 center（候选人不需要质心补偿）。
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
            print(f"C{i}: {counts.get(i, 0)}")

    def plot(self):
        """
        将主要选民群体和候选人绘制出来，剔除极端离群选民：
          1. 先算出选民重心 voter_center = np.mean(self.voters, axis=0)。
          2. 计算每个选民到 voter_center 的距离 d_i，取 95% 分位数 R95。
          3. 只把满足 d_i ≤ R95 的选民画出来，并用 R95 + margin 作为坐标轴范围。
          4. 支持者按不同候选人分配不同颜色（使用 matplotlib 的 tab 系列调色盘，确保不重复）。
          5. 赢家候选人用金色 (gold) 标记，其它候选人用红色 (red)。
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

        # （E）为每个候选人分配一种不重复的颜色
        # 使用 tab10/ tab20 等调色盘，根据候选人数量自动选择
        cmap_name = 'tab10' if self.num_candidates <= 10 else 'tab20'
        cmap = plt.cm.get_cmap(cmap_name, self.num_candidates)
        # 得到一个 (num_candidates, 4) 形状的颜色数组
        cand_colors = [cmap(i) for i in range(self.num_candidates)]
        # 每个选民的颜色 = 支持的候选人对应的颜色
        voter_colors = [cand_colors[v] for v in self.votes]

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

        # （G）绘制候选人：赢家金色、其它红色
        for idx, cand in enumerate(self.candidates):
            if idx == winner:
                color = 'gold'
                edgecolor = 'black'
                size = 180
            else:
                color = 'red'
                edgecolor = 'black'
                size = 150
            plt.scatter(cand[0], cand[1],
                        c=color,
                        edgecolors=edgecolor,
                        marker='X',
                        s=size,
                        linewidths=1.2,
                        zorder=5)
            # 在点上写上 “C0, C1, ...”
            plt.text(cand[0], cand[1], f'C{idx}',
                     fontsize=12, ha='center', va='center',
                     color='white',
                     bbox=dict(facecolor=color,
                               edgecolor='black',
                               boxstyle='circle,pad=0.3'),
                     zorder=6)

        # （H）设置坐标轴范围：以 voter_center 为中心，±(R95 + margin)
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

        例如，假设有一个随机漫步的策略函数：
            def random_walk(sim, step_size=1.0):
                sim.voters += np.random.normal(scale=step_size, size=sim.voters.shape)

        调用：sim.execute(random_walk, step_size=0.5)
        """
        if change_func is not None:
            # 只要传入的函数符合接口，就可以让选民更新
            change_func(self, **kwargs)
        # 如果 change_func=None，则什么都不做（选民保持原位）

    def propose(self):
        """
        根据当前选民的质心作为“新一轮候选人”的中心，重新生成 num_candidates 个候选人，
        并重置 self.votes = None，以便进行下一轮投票。
        """
        # 1. 计算当前选民质心
        voter_center = np.mean(self.voters, axis=0)
        # 2. 以质心为中心、spread_radius 为半径，重新生成候选人
        self.candidates = self._generate_candidates(voter_center)
        # 3. 重置之前的投票结果
        self.votes = None

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
