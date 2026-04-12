"""
MarketMamba V5.5 — 型態偵測函數庫
支援六大高勝率結構：W底/破底翻/M頭/頭肩底/收斂三角/上飄旗形
含：動態傾斜頸線、1/2~3/4 時間密碼、等幅測距
"""

import numpy as np
from scipy.signal import argrelextrema


# ==========================================
# 幾何與趨勢線運算核心
# ==========================================
def get_extrema(prices: np.ndarray, order: int) -> tuple:
    """用 scipy 尋找區域極小/極大值"""
    mins = argrelextrema(prices, np.less, order=order)[0]
    maxs = argrelextrema(prices, np.greater, order=order)[0]
    return mins, maxs


def get_trendline_val(x1: int, y1: float, x2: int, y2: float,
                      target_x: int) -> float:
    """計算兩點形成的趨勢線在 target_x 時的 Y 值 (動態頸線)"""
    if x2 == x1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (target_x - x1)


# ==========================================
# 🟡 模式 A: 標準 W底 / 多重底 (支援傾斜頸線)
# ==========================================
def detect_standard_w(prices: np.ndarray, mins: np.ndarray,
                      maxs: np.ndarray) -> tuple | None:
    """
    標準 W 底偵測
    條件：左右腳等高 (±3%)，當前價格接近動態頸線 (98%~105%)
    """
    if len(mins) < 2 or len(maxs) < 2:
        return None

    l1_idx, l2_idx = mins[-2], mins[-1]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    if h2_idx < l1_idx:
        return None

    current_idx = len(prices) - 1
    current = prices[-1]

    # 左右腳等高判定
    if not (0.97 <= l2 / l1 <= 1.03):
        return None

    # 動態頸線 (兩高點連線投影)
    dynamic_neckline = get_trendline_val(h1_idx, h1, h2_idx, h2, current_idx)
    if dynamic_neckline <= 0:
        return None

    ratio = current / dynamic_neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - (l2 / l1)) * 500 - abs(1 - ratio) * 300
        target = dynamic_neckline + (dynamic_neckline - min(l1, l2))
        return '🟡 標準W底', score, dynamic_neckline, target

    return None


# ==========================================
# 🟢 模式 B: 破底翻 (假跌破強勢站回)
# ==========================================
def detect_spring_w(prices: np.ndarray, mins: np.ndarray,
                    maxs: np.ndarray) -> tuple | None:
    """
    破底翻偵測 (Spring / 假跌破)
    條件：跌破左腳 2%~10% 後站回，爆發力最強的做多型態
    """
    if len(mins) < 2 or len(maxs) < 1:
        return None

    l1_idx, l2_idx = mins[-2], mins[-1]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    m_max = [x for x in maxs if l1_idx < x < l2_idx]
    if not m_max:
        return None

    neckline = prices[m_max[np.argmax(prices[m_max])]]
    current = prices[-1]

    # 破底定義：跌破左腳 2% ~ 10%
    if not (0.90 <= l2 / l1 <= 0.98):
        return None
    if current < l1:  # 必須站回左腳之上
        return None

    ratio = current / neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - ratio) * 400
        target = neckline + (neckline - l2)
        return '🟢 破底翻', score, neckline, target

    return None


# ==========================================
# 🔴 模式 C: M頭 (容許極端假突破誘多)
# ==========================================
def detect_m_top(prices: np.ndarray, mins: np.ndarray,
                 maxs: np.ndarray) -> tuple | None:
    """
    M 頭偵測 (False Breakout)
    條件：雙高等高 (右頭可高出左頭達 15% 的誘多假突破)
    """
    if len(maxs) < 2 or len(mins) < 1:
        return None

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    m_min = [x for x in mins if h1_idx < x < h2_idx]
    if not m_min:
        return None

    neckline = prices[m_min[np.argmin(prices[m_min])]]
    current = prices[-1]

    # 假突破定義：右頭可以高出左頭高達 15%
    if not (0.95 <= h2 / h1 <= 1.15):
        return None

    ratio = current / neckline
    if 0.95 <= ratio <= 1.02:
        score = 100 - abs(1 - ratio) * 300
        target = max(neckline - (max(h1, h2) - neckline), 0.1)
        return '🔴 M頭(偏空)', score, neckline, target

    return None


# ==========================================
# 🟣 模式 D: 頭肩底 (支援傾斜頸線)
# ==========================================
def detect_hns_bottom(prices: np.ndarray, mins: np.ndarray,
                      maxs: np.ndarray) -> tuple | None:
    """
    頭肩底偵測 (Head & Shoulders Bottom)
    條件：三低中間最深 (-2%)，兩肩等高 (±8%)
    """
    if len(mins) < 3 or len(maxs) < 2:
        return None

    l1_idx, l2_idx, l3_idx = mins[-3], mins[-2], mins[-1]
    l1, l2, l3 = prices[l1_idx], prices[l2_idx], prices[l3_idx]

    m_max1 = [x for x in maxs if l1_idx < x < l2_idx]
    m_max2 = [x for x in maxs if l2_idx < x < l3_idx]
    if not m_max1 or not m_max2:
        return None

    h1_idx = m_max1[np.argmax(prices[m_max1])]
    h2_idx = m_max2[np.argmax(prices[m_max2])]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    current_idx = len(prices) - 1
    current = prices[-1]

    # 頭部必須比兩肩低 2%
    if not (l2 < l1 * 0.98 and l2 < l3 * 0.98):
        return None
    # 兩肩等高 (±8%)
    if not (0.92 <= l1 / l3 <= 1.08):
        return None

    # 動態傾斜頸線
    dynamic_neckline = get_trendline_val(h1_idx, h1, h2_idx, h2, current_idx)

    ratio = current / dynamic_neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - (l1 / l3)) * 300 - abs(1 - ratio) * 300
        target = dynamic_neckline + (dynamic_neckline - l2)
        return '🟣 頭肩底', score, dynamic_neckline, target

    return None


# ==========================================
# 🔵 模式 E: 收斂三角形 (1/2 ~ 3/4 時間密碼)
# ==========================================
def detect_triangle(prices: np.ndarray, mins: np.ndarray,
                    maxs: np.ndarray) -> tuple | None:
    """
    收斂三角形偵測 (Converging Triangle)
    條件：高點下壓 + 低點上抬，在 50%~80% 進度突破壓力線
    """
    if len(mins) < 2 or len(maxs) < 2:
        return None

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    l1_idx, l2_idx = mins[-2], mins[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    current_idx = len(prices) - 1
    current = prices[-1]

    # 高點下壓 + 低點上抬
    if not (h1 > h2 * 1.01 and l1 < l2 * 0.99):
        return None

    # 計算 Apex 交點
    m_up = (h2 - h1) / (h2_idx - h1_idx) if h2_idx != h1_idx else 0
    m_low = (l2 - l1) / (l2_idx - l1_idx) if l2_idx != l1_idx else 0

    if m_up >= m_low:  # 確保線條收斂
        return None

    b_up = h1 - m_up * h1_idx
    b_low = l1 - m_low * l1_idx
    apex_x = (b_low - b_up) / (m_up - m_low)

    start_x = min(h1_idx, l1_idx)
    total_len = apex_x - start_x
    if total_len <= 0:
        return None

    # 時間密碼：50% ~ 80% 之間突破
    progress = (current_idx - start_x) / total_len
    if not (0.50 <= progress <= 0.80):
        return None

    # 動態壓力線
    resistance = m_up * current_idx + b_up

    if 0.99 <= current / resistance <= 1.05:
        score = 100 - abs(1 - current / resistance) * 400
        target = resistance + (h1 - l1)
        return '🔵 收斂三角', score, resistance, target

    return None


# ==========================================
# 💀 模式 F: 上飄旗形跌破 (波段跌幅測距)
# ==========================================
def detect_bear_flag(prices: np.ndarray, mins: np.ndarray,
                     maxs: np.ndarray) -> tuple | None:
    """
    上飄旗形跌破偵測 (Bear Flag)
    條件：主跌段 >10%，反彈不超過 2/3 後跌破旗形底部
    """
    if len(maxs) < 2 or len(mins) < 1:
        return None

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    # 抓取主跌段最低點
    m_min = [x for x in mins if h1_idx < x < h2_idx]
    if not m_min:
        return None

    l1_idx = m_min[np.argmin(prices[m_min])]
    l1 = prices[l1_idx]
    current = prices[-1]

    # 條件1: 主跌段很深 (跌幅大於 10%)
    if l1 > h1 * 0.90:
        return None
    # 條件2: 反彈形成上飄旗形 (反彈不超過主跌段的 2/3)
    if not (l1 < h2 < l1 + (h1 - l1) * 0.66):
        return None

    # 跌破旗形底部
    if current <= l1 * 1.02:
        score = 80
        target = max(h2 - (h1 - l1), 0.1)
        return '💀 上飄旗跌破', score, l1, target

    return None
