import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

class FastMarchingPointCloud:
    """针对点云优化的Fast Marching算法实现"""
    
    def __init__(self, k_neighbors=15):
        """
        初始化点云Fast Marching工具
        
        参数:
            k_neighbors: 用于构建k近邻图的邻居数量
        """
        self.vertices = None
        self.k_neighbors = k_neighbors
        self.graph = None
        self.distance = None
        self.seed_points = []
        self.max_distance = float('inf')
    
    def set_pointcloud(self, points, k_neighbors=None):
        """
        设置点云数据并构建k近邻图
        
        参数:
            points: 点云坐标数组 (n x 3)
            k_neighbors: k近邻数量（可选，覆盖初始化参数）
        """
        self.vertices = np.asarray(points, dtype=np.float64)
        
        if k_neighbors is not None:
            self.k_neighbors = k_neighbors
        
        # 构建k近邻图
        self._build_knn_graph()
        return True
    
    def _build_knn_graph(self):
        """使用k近邻构建点云的邻接图"""
        n_points = len(self.vertices)
        
        # 使用sklearn的KNN算法
        print(f"构建k={self.k_neighbors}近邻图...")
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, 
                                algorithm='kd_tree',
                                n_jobs=-1).fit(self.vertices)
        
        # 获取k近邻
        distances, indices = nbrs.kneighbors(self.vertices)
        
        # 构建对称的稀疏图（排除自身）
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([dist, dist])
        
        # 创建稀疏邻接矩阵
        self.graph = sparse.csr_matrix((data, (rows, cols)), 
                                       shape=(n_points, n_points))
        
        # 去除重复边（保留最小距离）
        self.graph = self.graph.minimum(self.graph.T)
        
        print(f"图构建完成：{n_points} 个点，{self.graph.nnz} 条边")
    
    def compute_distance(self, seed_points, max_distance=float('inf')):
        """
        使用Dijkstra算法计算从种子点到所有点的测地线距离
        
        参数:
            seed_points: 种子点索引列表
            max_distance: 最大搜索距离
        """
        self.seed_points = seed_points
        self.max_distance = max_distance
        
        if len(seed_points) == 0:
            print("错误：没有提供种子点")
            return False
        
        if self.graph is None:
            print("错误：图未构建，请先调用set_pointcloud")
            return False
        
        # 初始化距离数组
        self.distance = np.full(len(self.vertices), np.inf, dtype=np.float64)
        
        # 对每个种子点计算最短路径
        for seed in seed_points:
            dist_matrix = dijkstra(self.graph, 
                                  indices=[seed], 
                                  limit=max_distance,
                                  directed=False)
            # 更新为最小距离
            self.distance = np.minimum(self.distance, dist_matrix[0])
        
        return True
    
    def farthest_point_sampling(self, num_samples, initial_point=None):
        """
        使用最远点采样算法选择点
        
        参数:
            num_samples: 需要采样的点数
            initial_point: 初始点索引（可选）
        
        返回:
            samples: 采样点索引列表
            max_distance: 最大距离
        """
        n_points = len(self.vertices)
        
        if num_samples > n_points:
            print(f"警告：采样数量 {num_samples} 超过点数 {n_points}，将使用全部点")
            num_samples = n_points
        
        # 选择初始点
        if initial_point is None:
            initial_point = np.random.randint(0, n_points)
        
        samples = [initial_point]
        
        # 计算从初始点到所有点的距离
        self.compute_distance([initial_point])
        distances = self.distance.copy()
        
        # 迭代选择最远点
        print(f"开始最远点采样，目标数量: {num_samples}")
        for i in range(1, num_samples):
            if i % 10 == 0:
                print(f"  进度: {i}/{num_samples}")
            
            # 找到距离当前样本集最远的点
            next_point = np.argmax(distances)
            
            # 检查是否已经采样完所有可达点
            if np.isinf(distances[next_point]):
                print(f"警告：在采样 {i} 个点后，剩余点不可达")
                break
            
            samples.append(next_point)
            
            # 更新距离
            self.compute_distance([next_point])
            new_distances = self.distance.copy()
            distances = np.minimum(distances, new_distances)
        
        self.seed_points = samples
        max_dist = np.max(distances[np.isfinite(distances)])
        
        print(f"采样完成：{len(samples)} 个点，最大距离: {max_dist:.4f}")
        return samples, max_dist
    
    def estimate_optimal_k(self, sample_size=1000):
        """
        估算最优的k近邻数量
        
        参数:
            sample_size: 用于估算的样本数量
        
        返回:
            optimal_k: 建议的k值
        """
        n_points = len(self.vertices)
        sample_indices = np.random.choice(n_points, 
                                         min(sample_size, n_points), 
                                         replace=False)
        
        # 计算采样点的平均最近邻距离
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.vertices)
        distances, _ = nbrs.kneighbors(self.vertices[sample_indices])
        avg_nn_dist = np.mean(distances[:, 1])
        
        # 估算点密度
        volume = np.prod(np.ptp(self.vertices, axis=0))
        density = n_points / volume
        
        # 基于密度估算k值
        optimal_k = int(np.clip(15 * (density ** 0.33), 10, 50))
        
        print(f"点云统计：")
        print(f"  点数: {n_points}")
        print(f"  平均最近邻距离: {avg_nn_dist:.4f}")
        print(f"  估算最优k值: {optimal_k}")
        
        return optimal_k