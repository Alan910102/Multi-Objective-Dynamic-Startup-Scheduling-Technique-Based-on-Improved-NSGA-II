import random
import copy
import os
from datetime import datetime
import power
import numpy as np
import sys

MAX_START_TIME = 1000
GROUP_SIZE = 400
GENE_SIZE = 100
GENERATIONS = 500

TIME_STEP = 10
BUFFER_TIME = 10000
DEVICE_LIST = power.get_device_setup(GENE_SIZE)

TOURNAMENT_SIZE = 3

class Chromosome:
    """個體類別：處理單一排程（染色體）的資訊"""
    def __init__(self, gene_size, genes=None):
        self.gene_size = gene_size
        if genes is None:
            self.genes = []
            for i in range(gene_size):
                self.genes.append(random.uniform(0, 1000*gene_size))  
            random.shuffle(self.genes)
        else:
            self.genes = genes
        
        # 多目標優化使用的屬性
        self.objectives = ()       # 存放多個目標的分數 (time_score, hight_score, smooth_score)
        self.rank = 0              # Pareto 前沿層級 (0 為第一層)
        self.crowding_distance = 0 # 擁擠距離
        self.dominated_set = []    # 紀錄被該個體支配的集合 (Sp)
        self.domination_count = 0  # 紀錄該個體被多少人支配 (np)
    '''
    @staticmethod
    def device_type(gene_index):
        if gene_index < 10:
            return power.air_conditioner
        elif gene_index < 30:
            return power.printer
        else:
            return power.computer'''

    def calculate_objectives(self):
        """
        將原本的加權 fitness 拆分成獨立的多個目標：
        目標 1: 完工時間 (越小越好)
        目標 2: 功率峰值 (越小越好)
        目標 3: 功率平滑度 (越小越好)
        """
        def finish_time_score():
            if np.max(self.genes) > self.gene_size*MAX_START_TIME:   
                return 9999999999
            return np.sum(self.genes) + np.max(self.genes)*10
        
        def power_smooth_and_height_score():
            max_time = int(np.max(self.genes)+BUFFER_TIME)
            step = 10
            
            time_points = np.arange(0, max_time + 1, TIME_STEP)
            total_power_array = np.zeros(len(time_points))
            
            for i in range(self.gene_size):
                start_time = self.genes[i]
                device_cfg = DEVICE_LIST[i] # 取得該基因對應的機器設定
                
                # 計算這台機器需要多長的波形 (到排程結束為止)
                needed_duration = max_time - start_time
                
                if needed_duration > 0:
                    # 💡 呼叫 power.py 的智慧讀取
                    # 它會自動處理：讀取陣列 vs 動態計算，並完成抽樣(step)
                    profile_slice = power.get_smart_profile(
                        device_cfg, 
                        duration=needed_duration, 
                        step=TIME_STEP
                    )
                    
                    # 計算在總陣列中的起始位置
                    start_idx = int(start_time // TIME_STEP)
                    
                    
                    # 安全檢查：確保疊加時長度匹配
                    actual_len = min(len(profile_slice), len(total_power_array) - start_idx)
                    if actual_len > 0:
                        total_power_array[start_idx : start_idx + actual_len] += profile_slice[:actual_len]
            
            # 計算平滑度與峰值
            diff = np.diff( np.insert(total_power_array, 0, 0))
            smoothness = np.mean(diff ** 2)
            return np.max(total_power_array), smoothness
        
        time_score = finish_time_score()
        if time_score >= 9999999999:
            self.objectives = (9999999999, 9999999999, 9999999999)
        else:
            hight_score, smooth_score = power_smooth_and_height_score()
            self.objectives = (time_score, hight_score, smooth_score)
            
        return self.objectives

    def dominates(self, other):
        """
        判斷當前個體是否支配另一個體。
        條件：在所有目標上都不比對方差，且至少在一個目標上嚴格優於對方。(三個目標皆為越小越好)
        """
        and_condition = True
        or_condition = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                and_condition = False
                break
            elif self.objectives[i] < other.objectives[i]:
                or_condition = True
        return and_condition and or_condition

    def mutate(self, mutation_rate):
        def place_mutate():
            """
            兩台時間互換：隨機選擇兩個基因，將它們的值互換，並加上小幅隨機偏移。
            """
            for i in range(self.gene_size):
                if random.random()*2 < mutation_rate:
                    idx = random.randint(0, self.gene_size - 1)
                    # 先計算新的數值
                    new_val_i = self.genes[idx] + (random.random() - 0.5) * 10
                    new_val_idx = self.genes[i] + (random.random() - 0.5) * 10
                    # 限制最小值為 0，避免出現負數時間
                    self.genes[i] = max(0, new_val_i)
                    self.genes[idx] = max(0, new_val_idx)
        
        def cut_mutate():
            """
            區間偏移：隨機選擇一段基因區間，將該區間的所有基因值加上相同的隨機偏移量。
            """
            if random.random() < mutation_rate:
                sorted_genes = sorted(self.genes)            
                start=random.uniform(sorted_genes[0], sorted_genes[-1])
                end=random.uniform(sorted_genes[0], sorted_genes[-1])
                offset_target=[]
                if start > end:
                    for i in range(self.gene_size):
                        if self.genes[i] >= start or self.genes[i] <= end:
                            offset_target.append(i)
                else:
                    for i in range(self.gene_size):
                        if self.genes[i] >= start and self.genes[i] <= end:
                            offset_target.append(i)
                offset_value = random.uniform(sorted_genes[0], sorted_genes[-1])
                offset_value = (offset_value - 0.5 * (sorted_genes[-1] - sorted_genes[0]))
                if offset_target:
                    for i in offset_target:
                        new_value = self.genes[i] + offset_value
                        if new_value < 0:
                            self.genes[i] = self.genes[i] * 0.5 
                        else:
                            self.genes[i] = new_value
            
        if random.random() < 0.5:
            place_mutate()
        else:
            cut_mutate()

    def shake(self,mutation_rate=0.1):
        """
        基因微調：對每個基因以一定機率進行小幅隨機偏移
        """
        if random.random() < mutation_rate:
            for i in range(self.gene_size):
                if random.random() < 0.1:  
                    random_range = self.genes[i]*0.1
                    self.genes[i] += random.gauss(0, random_range / 2)  
                    if self.genes[i] < 0:
                        self.genes[i] = 0  


class GeneticAlgorithm:
    """演算法主控類別：多目標 NSGA-II 架構"""
    def __init__(self, group_size, gene_size, mutation_rate=0.1, add_genes=None):
        self.group_size = group_size
        self.gene_size = gene_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.population = []
        self.top_3_fronts = [] # 儲存前三層的 Pareto 前沿
        
        if add_genes is not None:
            print(f"正在加入 {len(add_genes)} 個指定基因到初始族群...")
            for genes in add_genes:
                if len(genes) != gene_size:
                    raise ValueError(f"基因長度必須為 {gene_size}，但收到 {len(genes)}")
                self.population.append(Chromosome(gene_size, genes))
            for _ in range(group_size-len(add_genes)):
                self.population.append(Chromosome(gene_size))
        else:
            self.population = [Chromosome(gene_size) for _ in range(group_size)]
            
        # 初始化第一代目標分數
        for p in self.population:
            p.calculate_objectives()

    def fast_non_dominated_sort(self, population):
        """快速非支配排序，將族群分層"""
        fronts = [[]]
        for p in population:
            p.dominated_set = []
            p.domination_count = 0
            for q in population:
                if p.dominates(q):
                    p.dominated_set.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
                        
            if len(next_front) == 0:
                break
            
            i += 1
            fronts.append(next_front)
        
        return fronts

    def calculate_crowding_distance(self, front):
        """計算擁擠距離以保持多樣性"""
        if not front:
            return
        num_objectives = len(front[0].objectives)
        for p in front:
            p.crowding_distance = 0
            
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])
            # 邊界點設為無限大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            min_obj = front[0].objectives[m]
            max_obj = front[-1].objectives[m]
            
            if max_obj - min_obj == 0:
                continue
                
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (max_obj - min_obj)

    def selection(self):
        """擁擠度錦標賽選擇 (Crowded-Comparison Tournament Selection)"""
        tournament_size = max(TOURNAMENT_SIZE , 2)
        competitors = random.sample(self.population, tournament_size)
        
        # NSGA-II 選擇邏輯：Rank 越低越好 (優先)，Rank 相同時擁擠距離越大越好
        best_competitor = min(competitors, key=lambda x: (x.rank, -x.crowding_distance))
        return best_competitor

    def crossover(self, parent1, parent2, avg_genes_value):
        child1_genes = [None] * self.gene_size
        child2_genes = [None] * self.gene_size

        def two_point_Crossover():
            """
            兩點交配：隨機選擇一段基因區間，將該區間的基因從一個父代複製到子代，其他部分從另一個父代複製。
            因為是對區間交配，所以會記錄被選中區間的基因索引，確保兩個子代在該區間內基因來自同一父代，其他部分來自另一父代。
            """
            def cut_chromosome(parent1, parent2):   #隨機選擇一個父代，從該父代的基因值分布中隨機選擇一段區間，並返回該區間的基因索引以及兩個父代
                p1_or_p2 = random.choice([parent1, parent2])
                another_parent = parent1 if p1_or_p2 == parent2 else parent2
                min_gene = min(p1_or_p2.genes)
                max_gene = max(p1_or_p2.genes)
                start=random.uniform(min_gene, max_gene)
                end=random.uniform(min_gene, max_gene)

                select_genes=[]
                if start > end:                    
                    for i in range(p1_or_p2.gene_size):
                        if p1_or_p2.genes[i]>=start or p1_or_p2.genes[i]<=end:
                            select_genes.append(i)
                else:
                    for i in range(p1_or_p2.gene_size):
                        if p1_or_p2.genes[i]>=start and p1_or_p2.genes[i]<=end:
                            select_genes.append(i)
                for i in range(self.gene_size-len(select_genes)):
                    select_genes.append(-1)
                return select_genes, p1_or_p2, another_parent

            select_genes = []
            timer=0
            while select_genes == [] or select_genes == list(range(self.gene_size)):  
                if timer > 200:
                    print(f"警告：切割點選擇失敗，已達最大嘗試次數。返回原始父代。 代數{self.generation}\n父代1基因: {parent1.genes}\n父代2基因: {parent2.genes}")
                    return parent1.genes, parent2.genes
                    raise ValueError("無法找到合適的切割點，請檢查基因內容或增加嘗試次數")
                select_genes, p1_or_p2, another_parent = cut_chromosome(parent1, parent2)
                timer += 1
                
            select_genes_sorted = sorted(select_genes, key=lambda idx: p1_or_p2.genes[idx])
            #print(f"{len(select_genes_sorted)}個 sorted選擇的基因索引: {select_genes_sorted}")  # Debug: 顯示被選中的基因索引排序結果 sord by 基因的時間
            #print(f"{len(select_genes)}個 選擇的基因索引: {select_genes}")  # Debug: 顯示被選中的基因索引
            j=0
            for i in range(self.gene_size):  #根據選擇的基因索引，將基因從對應的父代複製到子代
                if i != select_genes[j]:                #如果當前索引不在選擇的基因索引中，則從另一個父代複製基因
                    child1_genes[i]=p1_or_p2.genes[i]
                else:                                   #如果當前索引在選擇的基因索引中，則從被選中的父代複製基因
                    child1_genes[i]=another_parent.genes[select_genes_sorted[j]] 
                    j += 1
            j=0     
            k=0   
            for i in range(self.gene_size):  #反過來造child2，選擇的基因索引從被選中的父代複製，其他部分從另一個父代複製
                if i != select_genes[j]:
                    while k in select_genes:
                        k += 1
                    child2_genes[i]=another_parent.genes[k]
                    k += 1
                else:
                    child2_genes[i]=p1_or_p2.genes[select_genes[j]]
                    j += 1
            return child1_genes, child2_genes
            
        def arithmetic_crossover(avg_all_genes_value=avg_genes_value):  #基因平均值交配，兩個子代的基因為父代基因的平均值加上隨機偏移
            random_range = max(avg_all_genes_value * 0.1, avg_all_genes_value * (1 - self.generation / GENERATIONS))
            for i in range(self.gene_size):
                if random.random() < 0.5:  
                    offset1 = random.gauss(0, random_range / 3)  
                    offset2 = random.gauss(0, random_range / 3)
                    child1_genes[i] = max(0,(parent1.genes[i] + parent2.genes[i]) / 2 + offset1)
                    child2_genes[i] = max(0,(parent1.genes[i] + parent2.genes[i]) / 2 + offset2)
                else:
                    child1_genes[i] = parent1.genes[i]
                    child2_genes[i] = parent2.genes[i]
            return child1_genes, child2_genes
        
        if random.random() < min((self.generation/GENERATIONS),0.8):  
            child1_genes, child2_genes = arithmetic_crossover()
        else:
            child1_genes, child2_genes = two_point_Crossover()
        
        return Chromosome(self.gene_size, child1_genes), Chromosome(self.gene_size, child2_genes)

    def evolve(self):
        """執行一代的進化 (NSGA-II 主邏輯)"""
        # 1. 產生子代 (Q_t)
        offspring_population = []
        avg_genes_value = sum(gene for p in self.population for gene in p.genes) / (self.group_size * self.gene_size)
        
        while len(offspring_population) < self.group_size:
            p1 = self.selection()
            p2 = self.selection()   
            c1, c2 = self.crossover(p1, p2, avg_genes_value)
            c1.mutate(self.mutation_rate)
            c2.mutate(self.mutation_rate)
            c1.shake(self.mutation_rate)    #第二種種突變方式
            c2.shake(self.mutation_rate)
            # 計算子代的目標分數
            c1.calculate_objectives()
            c2.calculate_objectives()
            offspring_population.extend([c1, c2])
            
        # 截斷多餘的子代
        offspring_population = offspring_population[:self.group_size]
        
        # 2. 合併父代與子代 (R_t = P_t + Q_t)
        combined_population = self.population + offspring_population
        
        # 3. 進行非支配排序
        fronts = self.fast_non_dominated_sort(combined_population)
        
        # 存下此代的前三層前沿，供你後續取出畫圖用
        self.top_3_fronts = fronts[:3]
        
        # 4. 挑選進入下一代的個體 (P_{t+1})
        new_population = []
        front_idx = 0
        
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.group_size:
            # 整個前沿皆可放入
            self.calculate_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1
            
        # 若仍有空位，拿剩餘前沿中擁擠距離最大的填補
        if len(new_population) < self.group_size and front_idx < len(fronts):
            self.calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            new_population.extend(fronts[front_idx][:self.group_size - len(new_population)])
            
        self.population = new_population
        self.generation += 1


if __name__ == "__main__":
    if len(sys.argv) == 3:
        GROUP_SIZE, GENE_SIZE = int(sys.argv[1]), int(sys.argv[2])

    ga = GeneticAlgorithm(GROUP_SIZE, GENE_SIZE, mutation_rate=0.5)

    print(f"開始執行 NSGA-II... (G={GENERATIONS}, Pop={GROUP_SIZE}, Gene={GENE_SIZE})")
    
    for gen in range(GENERATIONS):
        ga.evolve()
        if gen % 10 == 0:
            print(f"第 {gen} 代, Rank0 解數量: {len(ga.top_3_fronts[0]) if ga.top_3_fronts else 0}")

    # --- 儲存前三層前沿結果 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_代數{GENERATIONS}_人口{GROUP_SIZE}_基因大小{GENE_SIZE}_錦標賽規模{TOURNAMENT_SIZE}"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(min(3, len(ga.top_3_fronts))):
        file_path = os.path.join(folder_name, f"rank{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for chromo in ga.top_3_fronts[i]:
                # 格式: 目標分數 | 基因
                obj_str = ", ".join([f"{obj:.2f}" for obj in chromo.objectives])
                gene_str = ", ".join([f"{g:.2f}" for g in chromo.genes])
                f.write(f"{obj_str} | {gene_str}\n")
    
    print(f"前三層前沿解已存入資料夾: {folder_name}")
    
    #import viewer
    #viewer.main()