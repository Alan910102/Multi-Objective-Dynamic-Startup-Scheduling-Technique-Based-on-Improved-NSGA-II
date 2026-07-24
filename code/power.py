import math
import numpy as np
import time

# 設定預計算的上限（50萬秒）
MAX_TIME = 100 * 5000

# ==========================================
# 1. 基礎功耗數學模型 (Pre-functions)
# ==========================================

def pre_large_ac(x):
    """大冷氣：基礎衰減 + 壓縮機週期性起伏"""
    base_decay = 1000 + 4000 * math.exp(-0.0005 * x)
    compressor_cycle = 300 * math.sin(0.001 * x)
    return base_decay + compressor_cycle

def pre_small_ac(x):
    """小冷氣：功率約為大冷氣的 0.6 倍"""
    base_decay = 1000 + 4000 * math.exp(-0.0005 * x)
    compressor_cycle = 300 * math.sin(0.002 * x)
    return (base_decay + compressor_cycle)* 0.6

def pre_computer(x):
    """電腦：開機衝擊 + 長期穩定負載 + CPU/GPU 波動"""
    startup_spike = 600 * math.exp(-0.005 * x)
    background_load = 200 * math.exp(-0.001 * x)
    fluctuation = 80 * math.sin(0.002 * x) * math.cos(0.005 * x)
    return 150 + startup_spike + background_load + fluctuation

def pre_printer(x):
    """印表機：預熱期、列印期、待機降溫期"""
    if x <= 2000:
        return 800 * (1 - math.exp(-0.002 * x))
    elif x <= 8000:
        return 600 + 200 * math.sin(0.008 * x)
    else:
        return 50 + 550 * math.exp(-0.001 * (x - 8000))

def pre_rotor(x):
    """轉子機器：高頻震盪馬達"""
    return 1200 + 400 * math.sin(0.05 * x)

def pre_furnace(x):
    """工業火爐：高能耗爬升"""
    return 3000 * (1 - math.exp(-0.0008 * x))


def simple1(x):
    return 1500


def simple2(x):
    if x < 20000:
        return 250
    else:
        return 0


def simple3(x):
    if x < 750:
        return 2000
    else:
        return 0

# ==========================================
# 2. 預計算 Numpy 陣列 (啟動時執行一次)
# ==========================================
print("[power.py] 正在預計算功耗波形陣列...")
start_pre = time.time()

# 產生地圖，方便後續管理
# 這裡先產生 1 秒 1 點的完整陣列
large_ac_profile_np = np.array([pre_large_ac(x) for x in range(MAX_TIME)])
small_ac_profile_np = np.array([pre_small_ac(x) for x in range(MAX_TIME)])
computer_profile_np = np.array([pre_computer(x) for x in range(MAX_TIME)])
printer_profile_np  = np.array([pre_printer(x) for x in range(MAX_TIME)])
rotor_profile_np    = np.array([pre_rotor(x) for x in range(MAX_TIME)])
furnace_profile_np  = np.array([pre_furnace(x) for x in range(MAX_TIME)])

print(f"[power.py] 預計算完成，耗時 {time.time() - start_pre:.2f} 秒")

# ==========================================
# 3. 機台配置工廠 (這裡決定你的 100 台組合)
# ==========================================

# 依照你的需求設定
DEVICE_CONFIG_LIST = [
    {"type": "Large AC", "count": 10,  "profile": large_ac_profile_np, "func": pre_large_ac},
    {"type": "Small AC", "count": 5,  "profile": small_ac_profile_np, "func": pre_small_ac},
    {"type": "Rotor",    "count": 0, "profile": rotor_profile_np,    "func": pre_rotor},
    {"type": "Furnace",  "count": 0, "profile": furnace_profile_np,  "func": pre_furnace},
    {"type": "Printer",  "count": 20,  "profile": printer_profile_np,  "func": pre_printer},
]
'''
DEVICE_CONFIG_LIST = [
    {"type": "Large AC", "count": 1,  "profile": large_ac_profile_np, "func": pre_large_ac},
    {"type": "Small AC", "count": 1,  "profile": small_ac_profile_np, "func": pre_small_ac},
    {"type": "Rotor",    "count": 1, "profile": rotor_profile_np,    "func": pre_rotor},
    {"type": "Furnace",  "count": 1, "profile": furnace_profile_np,  "func": pre_furnace},
    {"type": "Printer",  "count": 0,  "profile": printer_profile_np,  "func": pre_printer},
]'''


# 預設用來補足總數的機器
DEFAULT_CONFIG = {"type": "Computer", "profile": computer_profile_np, "func": pre_computer}

def get_device_setup(total_gene_size):
    """由 GA 呼叫，取得本次實驗的所有機台配置清單"""
    setup = []
    for cfg in DEVICE_CONFIG_LIST:
        setup.extend([cfg] * cfg["count"])
    
    remaining = total_gene_size - len(setup)
    if remaining > 0:
        setup.extend([DEFAULT_CONFIG] * remaining)
    elif remaining < 0:
        print(f"警告：設定機台數 ({len(setup)}) 超過 GENE_SIZE ({total_gene_size})！")
        setup = setup[:total_gene_size]
        
    return setup

# ==========================================
# 4. 智慧讀取函數 (防爆、防慢、防錯)
# ==========================================

def get_smart_profile(device_cfg, duration, step=10):
    """
    智慧讀取：
    1. 判斷 duration 是否超過預計算的 MAX_TIME。
    2. 如果超過，動態計算超出的部分並拼接。
    3. 按照 step (TIME_STEP) 進行抽樣回傳。
    """
    duration = int(duration)
    base_array = device_cfg["profile"]
    
    if duration <= MAX_TIME:
        # 情況 A：在預計算範圍內，直接切片 (速度最快)
        full_data = base_array[:duration]
    else:
        # 情況 B：超過範圍，拼接動態計算結果
        func = device_cfg["func"]
        extra_data = np.array([func(x) for x in range(MAX_TIME, duration)])
        full_data = np.concatenate((base_array, extra_data))
    
    # 根據 TIME_STEP 抽樣 (例如每 10 秒取一點)
    return full_data[::step]

# --- 以下保留舊有函數介面，確保相容性 ---
def air_conditioner(x):
    return large_ac_profile_np[int(x)] if x < MAX_TIME else pre_large_ac(x)

def computer(x):
    return computer_profile_np[int(x)] if x < MAX_TIME else pre_computer(x)

def printer(x):
    return printer_profile_np[int(x)] if x < MAX_TIME else pre_printer(x)