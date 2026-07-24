import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
import power

# 設定與 Pareto_ga.py 一致的常數
TIME_STEP = 10
BUFFER_TIME = 10000
# 是否在個體曲線上疊加極端情況（全部同時 / 全部分開）
PLT_EXT = False

def select_folder():
    """打開對話框讓使用者選擇結果資料夾"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="請選擇包含 rank.txt 的結果資料夾")
    return folder_path

def load_data(folder_path):
    """讀取資料夾內的 rank 檔案"""
    all_objectives = []
    all_genes = []
    all_ranks = []
    
    for rank in range(3):
        file_path = os.path.join(folder_path, f"rank{rank}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj_str, gene_str = line.strip().split(" | ")
                    objs = tuple(map(float, obj_str.split(", ")))
                    genes = list(map(float, gene_str.split(", ")))
                    
                    all_objectives.append(objs)
                    all_genes.append(genes)
                    all_ranks.append(rank)
                    
    return np.array(all_objectives), all_genes, np.array(all_ranks)

def calculate_power_curve(genes):
    """修正版：增加負數 start_time 防護與精確長度對齊"""
    genes_np = np.array(genes)
    gene_size = len(genes)
    # 確保 max_time 至少為 0
    max_genes = np.max(genes_np) if len(genes_np) > 0 else 0
    max_time = int(max(0, max_genes) + BUFFER_TIME)
    
    device_list = power.get_device_setup(gene_size)
    
    time_points = np.arange(0, max_time + 1, TIME_STEP)
    total_power_array = np.zeros(len(time_points))
    
    for i in range(gene_size):
        start_time = genes[i]
        device_cfg = device_list[i]
        
        # 1. 安全檢查：如果啟動時間超過繪圖範圍，直接跳過
        if start_time >= max_time:
            continue
            
        # 2. 計算所需的波形長度 (從啟動到結束)
        needed_duration = max_time - start_time
        
        if needed_duration > 0:
            # 取得原始波形
            profile_slice = power.get_smart_profile(
                device_cfg, 
                duration=needed_duration, 
                step=TIME_STEP
            )
            
            # 3. 核心修正：處理索引
            # 使用 // 取整數索引，並用 max(0, ...) 防止負數索引導致的切片錯誤
            start_idx = int(start_time // TIME_STEP)
            
            if start_idx < 0:
                # 如果啟動時間是負的，代表機器在 0 秒前就開了
                # 我們要跳過機器波形的前段，從對應 0 秒的位置開始疊加
                offset = abs(start_idx)
                profile_slice = profile_slice[offset:]
                start_idx = 0
            
            # 4. 再次確認長度，確保兩邊陣列形狀絕對一致
            # 剩餘空間 = 總陣列長度 - 啟動位置
            remaining_space = len(total_power_array) - start_idx
            # 實際能填入的長度為兩者最小值
            actual_len = min(len(profile_slice), remaining_space)
            
            if actual_len > 0:
                # 使用 [ : actual_len] 強制對齊雙方形狀
                total_power_array[start_idx : start_idx + actual_len] += profile_slice[:actual_len]
            
    return time_points, total_power_array

def plot_individual_curve(genes, objectives, x_max, y_max, plt_ext=False):
    """【更新】動態標註不同機台類型的啟動點"""
    time_points, power_array = calculate_power_curve(genes)
    time_points = np.insert(time_points, 0, -10)  # 在開頭插入 0 秒
    power_array = np.insert(power_array, 0, 0) 
    
    gene_size = len(genes)
    device_list = power.get_device_setup(gene_size)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_points, power_array, color='red', alpha=0.9, linewidth=3.5, label='Total Power')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max * 1.05)
    ax.set_aspect('auto')
    
    # 用來存放已標註的類型，避免圖例重複
    labeled_types = set()
    
    # 定義顏色與標記地圖 (可隨意增加)
    style_map = {
        "Large AC": {"color": "red", "marker": "^", "s": 50},
        "Small AC": {"color": "salmon", "marker": "v", "s": 40},
        "Rotor":    {"color": "green", "marker": "s", "s": 30},
        "Furnace":  {"color": "purple", "marker": "D", "s": 45},
        "Printer":  {"color": "blue", "marker": "P", "s": 40},
        "Computer": {"color": "orange", "marker": "o", "s": 20},
    }

    for i in range(gene_size):
        start_time = genes[i]
        dev_type = device_list[i]["type"]
        style = style_map.get(dev_type, {"color": "gray", "marker": "x", "s": 20})
        
        # 找出對應 Y 軸數值
        y_val = power_array[min(int(start_time // TIME_STEP), len(power_array)-1)]
        
        # 只在第一次出現該類型時添加 label
        label = dev_type if dev_type not in labeled_types else ""
        #ax.scatter(start_time, y_val, label=label, zorder=5, **style)
        labeled_types.add(dev_type)
    
    ax.set_title(f"Individual Schedule Details\nTime: {objectives[0]:.0f}, Peak: {objectives[1]:.0f}, Smooth: {objectives[2]:.0f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    # 若使用者要求，疊加極端情況曲線 (全部同時 / 全部分開)
    if plt_ext:
        device_list = power.get_device_setup(gene_size)
        # simultaneous
        genes_sim = [0] * gene_size
        t_sim, p_sim = calculate_power_curve(genes_sim)
        t_sim = np.insert(t_sim, 0, -10)
        p_sim = np.insert(p_sim, 0, 0)
        ax.plot(t_sim, p_sim, color='magenta', linestyle='--', alpha=0.9, linewidth=3.5, label='Extreme: All simultaneous')
        # sequential
        genes_seq = [i * 10000 for i in range(gene_size)]
        t_seq, p_seq = calculate_power_curve(genes_seq)
        t_seq = np.insert(t_seq, 0, -10)
        p_seq = np.insert(p_seq, 0, 0)
        ax.plot(t_seq, p_seq, color='cyan', linestyle='--', alpha=0.9, linewidth=3.5, label='Extreme: All sequential')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.show()

def main():
    # 使用檔內常數 PLT_EXT 控制是否疊加極端情況
    plt_ext = PLT_EXT

    folder_path = select_folder()
    if not folder_path: return

    objectives, genes, ranks = load_data(folder_path)
    if len(objectives) == 0: return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue']
    labels = ['Rank 0 (Pareto Frontier)', 'Rank 1', 'Rank 2']
    scatter_plots = []
    
    for r in range(3):
        mask = ranks == r
        if np.any(mask):
            objs_r = objectives[mask]
            sc = ax.scatter(objs_r[:, 0], objs_r[:, 1], objs_r[:, 2], 
                            c=colors[r], label=labels[r], s=50, alpha=0.7, picker=True, pickradius=5)
            scatter_plots.append((sc, np.where(mask)[0]))

    ax.set_xlabel('Objective 1: Time Score')
    ax.set_ylabel('Objective 2: Power')
    ax.set_zlabel('Objective 3: Smooth')
    ax.set_title('NSGA-II Optimization Results (Click a point to view details)')
    ax.legend()

    # 計算本次執行中統一使用的軸範圍
    global_max_time = int(max((max(g) if len(g) > 0 else 0) for g in genes) + BUFFER_TIME)
    global_max_power = float(np.max(objectives[:, 1])) if len(objectives) > 0 else 0.0

    def onpick(event):
        for sc, global_indices in scatter_plots:
            if event.artist == sc:
                ind = event.ind[0]
                global_idx = global_indices[ind]
                print(f"Showing details for individual {global_idx}...")
                plot_individual_curve(genes[global_idx], objectives[global_idx], global_max_time, global_max_power, plt_ext=plt_ext)
                break

    fig.canvas.mpl_connect('pick_event', onpick)
    import plotly.graph_objects as go
    web_fig = go.Figure()
    for r, color, label in zip(range(3), colors, labels):
        mask = ranks == r
        if np.any(mask):
            indices = np.where(mask)[0]
            web_fig.add_trace(go.Scatter3d(
                x=objectives[mask, 0],
                y=objectives[mask, 1],
                z=objectives[mask, 2],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.8),
                name=label,
                text=[f"Ind #{i} (Rank {r})" for i in indices],
                hoverinfo='text'
            ))

    web_fig.update_layout(
        title='NSGA-II Optimization Results (Hover for details)',
        scene=dict(
            xaxis_title='Objective 1: Time Score',
            yaxis_title='Objective 2: Power',
            zaxis_title='Objective 3: Smooth'
        ),
        legend=dict(itemsizing='constant')
    )
    web_fig.write_html('index.html')
    plt.show()

if __name__ == "__main__":
    main()