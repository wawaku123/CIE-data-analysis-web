import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import streamlit as st
import io
import base64
import chardet
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ====================== 1. 页面与中文配置（修复中文显示方框问题） ======================
st.set_page_config(
    layout="wide",
    page_title="CIE色点校正工具",
    page_icon="🔧"
)


# 修复matplotlib中文显示问题（跨平台兼容）
def setup_chinese_font():
    """配置matplotlib中文显示"""
    # 备选字体列表（覆盖Windows/macOS/Linux）
    font_list = [
        'Microsoft YaHei',  # Windows
        'SimHei',  # Windows
        'SimSun',  # Windows
        'WenQuanYi Micro Hei',  # Linux
        'PingFang SC',  # macOS
        'Heiti SC',  # macOS
        'Arial Unicode MS'  # 通用备选
    ]

    # 尝试设置字体
    for font_name in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # 验证字体是否可用
            fm.FontProperties(family=font_name)
            return
        except:
            continue

    # 如果都失败，使用默认字体+禁用中文检查
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# 初始化中文配置
setup_chinese_font()

# ====================== 2. 目标色区定义 ======================
TARGET_ZONES = {
    'DK32': [(0.2743, 0.265), (0.277, 0.27), (0.2825, 0.27), (0.2798, 0.265)],
    'DK33': [(0.2715, 0.26), (0.2743, 0.265), (0.2798, 0.265), (0.2771, 0.26)],
    'DK34': [(0.2688, 0.255), (0.2715, 0.26), (0.2771, 0.26), (0.2743, 0.255)],
    'DK35': [(0.2661, 0.25), (0.2688, 0.255), (0.2743, 0.255), (0.2716, 0.25)],
    'DL32': [(0.2798, 0.265), (0.2825, 0.27), (0.288, 0.27), (0.2853, 0.265)],
    'DL33': [(0.2771, 0.26), (0.2798, 0.265), (0.2853, 0.265), (0.2826, 0.26)],
    'DL34': [(0.2743, 0.255), (0.2771, 0.26), (0.2826, 0.26), (0.2799, 0.255)],
    'DL35': [(0.2716, 0.25), (0.2743, 0.255), (0.2799, 0.255), (0.2771, 0.25)]
}
TARGET_CENTER = (0.2771, 0.26)  # 目标中心点


# ====================== 3. 辅助函数 ======================
def detect_file_encoding(file_content):
    """检测文件编码"""
    result = chardet.detect(file_content[:10000])
    encoding = result['encoding']
    confidence = result['confidence']

    if encoding is None or confidence < 0.8:
        encoding = 'utf-8'

    if encoding.lower() in ['gb2312', 'gbk', 'gb18030', 'big5']:
        encoding = 'gbk'

    return encoding


def read_csv_file(uploaded_file):
    """读取CSV文件，自动检测编码"""
    file_content = uploaded_file.getvalue()
    encoding = detect_file_encoding(file_content)

    # 尝试多种编码
    encodings_to_try = [encoding, 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin1']

    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            st.info(f"使用编码: {enc}")
            return df
        except UnicodeDecodeError:
            continue

    # 如果都失败，使用latin1
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.warning("使用latin1编码，中文字符可能显示异常")
    return df


def point_in_polygon(point, polygon):
    """判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            x_intersect = (y - yi) * (xj - xi) / (yj - yi + 1e-8) + xi
            if x <= x_intersect:
                inside = not inside
    return inside


def point_in_any_zone(point):
    """判断点是否在任何目标色区内"""
    for zone in TARGET_ZONES.values():
        if point_in_polygon(point, zone):
            return True
    return False


def calculate_zone_ratio(points):
    """计算目标色区占比"""
    if len(points) == 0:
        return 0.0
    in_zone_count = sum(1 for point in points if point_in_any_zone(point))
    return in_zone_count / len(points)


def calculate_original_color_center(df, bin_code_col='bin_code', ciex_col='ciex', ciey_col='ciey'):
    """计算原始色坐标中心（排除未点亮、VF1不良、ciex=0、ciey=0的所有有效测试色点）"""
    # 关键修改1：使用数据副本，避免修改原数据
    df_copy = df.copy()

    if bin_code_col not in df_copy.columns or ciex_col not in df_copy.columns or ciey_col not in df_copy.columns:
        return None

    # 排除条件：
    # 1. 未点亮
    # 2. VF1不良
    # 3. 色坐标为0的点
    exclude_bins = ['未点亮', 'VF1不良']
    valid_mask = ~df_copy[bin_code_col].isin(exclude_bins)

    # 排除ciex=0或ciey=0的点
    valid_mask &= (df_copy[ciex_col] != 0) & (df_copy[ciey_col] != 0)
    valid_mask &= (df_copy[ciex_col] > 0.1) & (df_copy[ciey_col] > 0.1)  # 同时排除异常小值

    valid_df = df_copy[valid_mask]

    if len(valid_df) == 0:
        st.warning("没有找到有效的色坐标数据（所有数据都被排除）")
        return None

    # 计算中心点
    center_x = valid_df[ciex_col].mean()
    center_y = valid_df[ciey_col].mean()

    # 计算被排除的数据数量
    excluded_count = len(df_copy) - len(valid_df)
    vf1_excluded = len(df_copy[df_copy[bin_code_col] == 'VF1不良'])
    unlit_excluded = len(df_copy[df_copy[bin_code_col] == '未点亮'])
    zero_coord_excluded = len(df_copy[(df_copy[ciex_col] == 0) | (df_copy[ciey_col] == 0)])

    st.write(f"**数据筛选详情**:")
    st.write(f"- 总数据点: {len(df_copy)}")
    st.write(f"- 有效色点: {len(valid_df)} (用于中心点计算)")
    st.write(f"- 被排除的点: {excluded_count}")
    st.write(f"  - VF1不良: {vf1_excluded}")
    st.write(f"  - 未点亮: {unlit_excluded}")
    st.write(f"  - 色坐标为0: {zero_coord_excluded}")

    return (center_x, center_y)


def get_color_optimization_points(df, bin_code_col='bin_code', ciex_col='ciex', ciey_col='ciey'):
    """获取用于色坐标优化的基准点（排除未点亮、VF1不良、色坐标为0的所有有效色点）"""
    # 关键修改2：使用数据副本
    df_copy = df.copy()

    if bin_code_col not in df_copy.columns or ciex_col not in df_copy.columns or ciey_col not in df_copy.columns:
        return np.array([])

    # 排除条件：
    # 1. 未点亮
    # 2. VF1不良
    # 3. 色坐标为0的点
    exclude_bins = ['未点亮', 'VF1不良']
    valid_mask = ~df_copy[bin_code_col].isin(exclude_bins)

    # 排除ciex=0或ciey=0的点
    valid_mask &= (df_copy[ciex_col] != 0) & (df_copy[ciey_col] != 0)
    valid_mask &= (df_copy[ciex_col] > 0.1) & (df_copy[ciey_col] > 0.1)  # 同时排除异常小值

    valid_df = df_copy[valid_mask]

    if len(valid_df) == 0:
        return np.array([])

    return valid_df[[ciex_col, ciey_col]].values


def adjust_color_points_with_center_shift(points, original_center, target_center, concentration_gain=1.0,
                                          center_shift_gain=0.0):
    """
    调整色点集中度并移动中心点
    Args:
        points: 原始色点坐标数组 (n×2)
        original_center: 原始中心点
        target_center: 目标中心点
        concentration_gain: 集中度增益 (1.0-5.0)
        center_shift_gain: 中心点移动增益 (0.0-1.0)
    """
    if len(points) == 0 or original_center is None or target_center is None:
        return points.copy(), original_center

    # 1. 计算中心点移动向量
    center_vector = np.array(target_center) - np.array(original_center)

    # 2. 计算实际移动距离（基于center_shift_gain）
    # center_shift_gain=0: 不移动; center_shift_gain=1: 完全移动到目标中心
    actual_center = np.array(original_center) + center_vector * center_shift_gain

    # 3. 将点云平移到原始中心
    points_centered = points - original_center

    # 4. 应用集中度缩放
    if concentration_gain > 0:
        scale_factor = 1.0 / np.sqrt(concentration_gain)
        points_scaled = points_centered * scale_factor
    else:
        points_scaled = points_centered

    # 5. 将点云移动到实际中心位置
    adjusted_points = points_scaled + actual_center

    return adjusted_points, actual_center


def get_normal_dvf_range(df):
    """计算正常材料的DVF范围（排除不良）"""
    # 关键修改3：使用数据副本
    df_copy = df.copy()

    if 'dvf' not in df_copy.columns or 'bin_code' not in df_copy.columns:
        return -0.043, -0.045  # 默认值

    normal_mask = ~df_copy['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_df = df_copy[normal_mask]

    if len(normal_df) == 0 or len(normal_df['dvf'].dropna()) == 0:
        return -0.043, -0.045  # 默认值

    dvf_values = normal_df['dvf'].dropna()
    dvf_q1 = dvf_values.quantile(0.25)
    dvf_q3 = dvf_values.quantile(0.75)

    return dvf_q1, dvf_q3  # 返回四分位距


def modify_dvf_smart(df, modify_ratio=1.0):
    """智能修改DVF不良数据，贴近正常材料范围"""
    # 操作原数据（此处需要修改数据，保持原有逻辑）
    if 'dvf' not in df.columns or 'bin_code' not in df.columns:
        return df, 0, []

    mask = (df['bin_code'] == 'DVF不良')
    total_bad = mask.sum()

    if total_bad == 0:
        return df, 0, []

    # 计算实际修改数量
    modify_count = int(total_bad * modify_ratio)

    # 获取正常材料的DVF范围
    dvf_min, dvf_max = get_normal_dvf_range(df)

    # 获取正常材料的DVF值
    normal_mask = ~df['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_dvf_values = df[normal_mask]['dvf'].dropna().values

    # 如果正常材料有足够的数据，从中采样
    if len(normal_dvf_values) > 0:
        # 从正常材料的DVF值中采样
        if len(normal_dvf_values) >= modify_count:
            new_values = np.random.choice(normal_dvf_values, size=modify_count, replace=True)
        else:
            # 如果正常材料数据不足，从范围内生成
            new_values = np.random.uniform(dvf_min, dvf_max, size=modify_count)
    else:
        # 从范围内生成
        new_values = np.random.uniform(dvf_min, dvf_max, size=modify_count)

    # 获取要修改的索引
    bad_indices = df[mask].index.tolist()
    modify_indices = bad_indices[:modify_count]

    # 修改数据
    modification_details = []
    for i, idx in enumerate(modify_indices):
        original_value = df.at[idx, 'dvf']
        original_decimal = len(str(original_value).split('.')[1]) if '.' in str(original_value) else 3

        new_value = new_values[i] if i < len(new_values) else np.random.uniform(dvf_min, dvf_max)
        new_value = round(new_value, original_decimal)

        df.at[idx, 'dvf'] = new_value

        # 记录修改
        test_no_col = 'test_no' if 'test_no' in df.columns else 'TestNo' if 'TestNo' in df.columns else None
        test_no = df.at[idx, test_no_col] if test_no_col else idx

        modification_details.append({
            'test_no': test_no,
            '原值': original_value,
            '新值': new_value,
            '修改项': 'dvf',
            '小数位数': original_decimal
        })

    return df, modify_count, modification_details


def modify_vf2_smart(df, modify_ratio=1.0):
    """智能修改VF2不良数据，贴近正常材料范围"""
    # 操作原数据（此处需要修改数据，保持原有逻辑）
    vf2_cols = ['forward_voltage2_V', 'forward_voltage2', 'forward_vc', 'forward_vcf']
    vf2_col = None

    # 找到可用的VF2列
    for col in vf2_cols:
        if col in df.columns:
            vf2_col = col
            break

    if vf2_col is None or 'bin_code' not in df.columns:
        return df, 0, []

    mask = (df['bin_code'] == 'VF2不良')
    total_bad = mask.sum()

    if total_bad == 0:
        return df, 0, []

    # 计算实际修改数量
    modify_count = int(total_bad * modify_ratio)

    # 获取正常材料的VF2范围
    normal_mask = ~df['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_vf2_values = df[normal_mask][vf2_col].dropna().values

    if len(normal_vf2_values) > 0:
        vf2_min = np.percentile(normal_vf2_values, 25)
        vf2_max = np.percentile(normal_vf2_values, 75)

        # 从正常材料的VF2值中采样
        if len(normal_vf2_values) >= modify_count:
            new_values = np.random.choice(normal_vf2_values, size=modify_count, replace=True)
        else:
            new_values = np.random.uniform(vf2_min, vf2_max, size=modify_count)
    else:
        # 默认范围
        vf2_min, vf2_max = 4.7, 5.0
        new_values = np.random.uniform(vf2_min, vf2_max, size=modify_count)

    # 获取要修改的索引
    bad_indices = df[mask].index.tolist()
    modify_indices = bad_indices[:modify_count]

    # 修改数据
    modification_details = []
    for i, idx in enumerate(modify_indices):
        original_value = df.at[idx, vf2_col]
        original_decimal = len(str(original_value).split('.')[1]) if '.' in str(original_value) else 3

        new_value = new_values[i] if i < len(new_values) else np.random.uniform(vf2_min, vf2_max)
        new_value = round(new_value, original_decimal)

        df.at[idx, vf2_col] = new_value

        # 记录修改
        test_no_col = 'test_no' if 'test_no' in df.columns else 'TestNo' if 'TestNo' in df.columns else None
        test_no = df.at[idx, test_no_col] if test_no_col else idx

        modification_details.append({
            'test_no': test_no,
            '原值': original_value,
            '新值': new_value,
            '修改项': vf2_col,
            '小数位数': original_decimal
        })

    return df, modify_count, modification_details


def apply_color_adjustment_with_center_shift(df, original_center, target_center, concentration_gain=1.0,
                                             center_shift_gain=0.0, ciex_col='ciex', ciey_col='ciey'):
    """应用色坐标调整（包含中心点移动）"""
    # 操作原数据（此处需要修改数据，保持原有逻辑）
    if ciex_col not in df.columns or ciey_col not in df.columns or 'bin_code' not in df.columns:
        return df, [], None

    if original_center is None or target_center is None:
        st.error("无法计算中心点")
        return df, [], None

    # 获取色坐标不良的数据
    mask = df['bin_code'] == '色坐标不良'
    if not mask.any():
        return df, [], original_center

    # 提取原始点
    original_points = df.loc[mask, [ciex_col, ciey_col]].values

    # 调整集中度并移动中心点
    adjusted_points, actual_center = adjust_color_points_with_center_shift(
        original_points,
        original_center,
        target_center,
        concentration_gain,
        center_shift_gain
    )

    # 标准化到4位小数
    adjusted_points = np.round(adjusted_points, 4)

    # 记录修改详情
    modification_details = []
    for i, idx in enumerate(df[mask].index):
        original_x = df.at[idx, ciex_col]
        original_y = df.at[idx, ciey_col]
        new_x = adjusted_points[i, 0]
        new_y = adjusted_points[i, 1]

        df.at[idx, ciex_col] = new_x
        df.at[idx, ciey_col] = new_y

        test_no_col = 'test_no' if 'test_no' in df.columns else 'TestNo' if 'TestNo' in df.columns else None
        test_no = df.at[idx, test_no_col] if test_no_col else idx

        modification_details.append({
            'test_no': test_no,
            '原ciex': original_x,
            '新ciex': new_x,
            '原ciey': original_y,
            '新ciey': new_y
        })

    return df, modification_details, actual_center


def plot_final_cie_chart(df, bin_col, ciex_col, ciey_col, target_center):
    """绘制最终的CIE坐标图（生成文件时展示）"""
    # 关键修改4：使用数据副本，彻底隔离原数据
    df_copy = df.copy()

    # 计算最终中心点（使用副本数据）
    final_center = calculate_original_color_center(df_copy, bin_col, ciex_col, ciey_col)
    if final_center is None:
        st.warning("无法计算最终色坐标中心")
        return

    # 过滤有效数据（基于副本）
    exclude_bins = ['未点亮', 'VF1不良']
    valid_mask = ~df_copy[bin_col].isin(exclude_bins)
    valid_mask &= (df_copy[ciex_col] != 0) & (df_copy[ciey_col] != 0) & (df_copy[ciex_col] > 0.1) & (
            df_copy[ciey_col] > 0.1)
    valid_df = df_copy[valid_mask].copy()  # 再次复制，确保不修改

    if len(valid_df) == 0:
        st.warning("无有效色坐标数据")
        return

    # 计算最终占比
    final_points = valid_df[[ciex_col, ciey_col]].values
    final_ratio = calculate_zone_ratio(final_points)

    # 绘制最终图表
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_ZONES)))

    # 绘制目标色区
    for (zone_name, zone_points), color in zip(TARGET_ZONES.items(), colors):
        polygon = Polygon(zone_points, closed=True, alpha=0.3, label=zone_name, color=color)
        ax.add_patch(polygon)

    # 绘制中心点
    ax.plot(final_center[0], final_center[1], 'mo', markersize=10, label='最终中心点', zorder=5)
    ax.plot(target_center[0], target_center[1], 'ro', markersize=8, label='目标中心点', zorder=5, alpha=0.7)

    # 绘制最终色点
    ax.scatter(valid_df[ciex_col], valid_df[ciey_col],
               s=20, alpha=0.6, c='darkred', label=f'最终色点 (n={len(valid_df)})')

    # 图表配置
    ax.set_xlabel('CIE X 坐标', fontsize=12)
    ax.set_ylabel('CIE Y 坐标', fontsize=12)
    ax.set_title(f'最终色点分布（目标色区占比: {final_ratio:.2%}）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 优化图例
    ax.legend(
        fontsize=9,
        loc='upper right',
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        handletextpad=0.2,
        labelspacing=0.2
    )

    st.pyplot(fig, use_container_width=True)
    st.info(f"最终目标色区占比: {final_ratio:.2%} | 最终中心点: ({final_center[0]:.4f}, {final_center[1]:.4f})")


def clear_cache_data():
    """清除缓存数据"""
    keys_to_clear = ['original_df', 'current_df']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ 缓存数据已清除！请重新上传文件")
    st.rerun()


# ====================== 4. 主程序 ======================
def main():
    st.title("🔧 CIE色点校正工具")
    st.markdown("上传LED分光数据文件，调整不良Bin，优化色坐标集中性")

    # 新增：清除缓存按钮（放在顶部）
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🗑️ 清除缓存数据", use_container_width=True, type='secondary'):
            clear_cache_data()

    # 文件上传
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

    if uploaded_file is not None:
        # 读取文件
        df = read_csv_file(uploaded_file)

        # 初始化session state
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()

        if 'current_df' not in st.session_state:
            st.session_state.current_df = df.copy()

        df = st.session_state.current_df

        # 显示列名信息
        st.subheader("📊 数据列名信息")
        st.write(f"总列数: {len(df.columns)}")
        st.write("**关键列识别:**")

        # 查找关键列
        key_columns_found = {}
        key_columns_map = {
            '测试编号': ['test_no', 'TestNo', 'Test_No'],
            'Bin代码': ['bin_code', 'BinCode', 'Bin_Code'],
            'DVF': ['dvf', 'DVF', 'ΔVF'],
            'VF2': ['forward_voltage2_V', 'forward_voltage2', 'forward_vc', 'forward_vcf'],
            'CIE x': ['ciex', 'CIE_x', 'x_coordinate'],
            'CIE y': ['ciey', 'CIE_y', 'y_coordinate']
        }

        for col_name, possible_names in key_columns_map.items():
            found_col = None
            for name in possible_names:
                if name in df.columns:
                    found_col = name
                    key_columns_found[col_name] = found_col
                    break

        for col_name, found_col in key_columns_found.items():
            st.write(f"✓ {col_name}: {found_col}")

        # 显示原始数据统计
        st.subheader("原始数据分析")
        st.write(f"总数据点数: {len(df)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Bin代码' in key_columns_found:
                bin_count = len(df[key_columns_found['Bin代码']].unique())
                st.metric("不同Bin数量", bin_count)

        with col2:
            st.metric("总行数", len(df))

        with col3:
            if 'Bin代码' in key_columns_found:
                bad_bins = ['未点亮', '色坐标不良', 'DVF不良', 'VF2不良']
                bad_count = df[df[key_columns_found['Bin代码']].isin(bad_bins)].shape[0]
                st.metric("不良品数量", bad_count)

        # 1. Bin统计
        st.subheader("1. Bin数量统计")
        if 'Bin代码' in key_columns_found:
            bin_col = key_columns_found['Bin代码']
            bin_stats = df[bin_col].value_counts().reset_index()
            bin_stats.columns = ['Bin Code', '数量']
            bin_stats['占比'] = (bin_stats['数量'] / len(df) * 100).round(2).astype(str) + '%'

            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(bin_stats, height=300, use_container_width=True)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                top_bins = bin_stats.head(10)
                ax.barh(top_bins['Bin Code'][::-1], top_bins['数量'][::-1])
                ax.set_xlabel('数量', fontsize=10)
                ax.set_title('Top 10 Bin分布', fontsize=12)
                st.pyplot(fig, use_container_width=True)

        # 2. 不良Bin校正
        st.subheader("2. 不良Bin校正")

        if 'Bin代码' in key_columns_found:
            bin_col = key_columns_found['Bin代码']
            col1, col2 = st.columns(2)

            with col1:
                st.write("**VF2不良校正**")
                vf2_ratio = st.slider("VF2校正比例", 0.0, 1.0, 1.0, 0.1, key="vf2_ratio")

                if st.button("校正VF2不良", use_container_width=True):
                    df, count, details = modify_vf2_smart(df, vf2_ratio)
                    st.session_state.current_df = df

                    if count > 0:
                        st.success(f"✅ 已校正 {count} 个VF2不良（比例: {vf2_ratio * 100:.1f}%）")
                        st.write("**前5条修改详情:**")
                        for detail in details[:5]:
                            st.write(
                                f"test_no {detail['test_no']}: {detail['修改项']} 从 {detail['原值']} 修改为 {detail['新值']}")
                    else:
                        st.info("没有找到需要校正的VF2不良")

            with col2:
                st.write("**DVF不良校正**")
                dvf_ratio = st.slider("DVF校正比例", 0.0, 1.0, 1.0, 0.1, key="dvf_ratio")

                if st.button("校正DVF不良", use_container_width=True):
                    df, count, details = modify_dvf_smart(df, dvf_ratio)
                    st.session_state.current_df = df

                    if count > 0:
                        st.success(f"✅ 已校正 {count} 个DVF不良（比例: {dvf_ratio * 100:.1f}%）")
                        st.write("**前5条修改详情:**")
                        for detail in details[:5]:
                            st.write(
                                f"test_no {detail['test_no']}: {detail['修改项']} 从 {detail['原值']} 修改为 {detail['新值']}")
                    else:
                        st.info("没有找到需要校正的DVF不良")

        # 3. 色坐标集中性调整（增强版：中心点移动+集中度增益）
        st.subheader("3. 色坐标集中性优化（中心点移动+集中度增益）")

        ciex_col = key_columns_found.get('CIE x')
        ciey_col = key_columns_found.get('CIE y')
        bin_col = key_columns_found.get('Bin代码')

        if ciex_col and ciey_col and bin_col:
            # 计算原始中心点（排除未点亮、VF1不良、色坐标(0,0)的所有有效测试色点）
            original_center = calculate_original_color_center(df, bin_col, ciex_col, ciey_col)
            target_center = TARGET_CENTER

            if original_center is not None:
                st.write(f"**原始色坐标中心**: ({original_center[0]:.4f}, {original_center[1]:.4f})")
                st.write(f"**目标色坐标中心**: ({target_center[0]:.4f}, {target_center[1]:.4f})")

                # 计算原始中心到目标的距离
                center_distance = np.sqrt(
                    (original_center[0] - target_center[0]) ** 2 + (original_center[1] - target_center[1]) ** 2)
                st.write(f"**原始中心到目标的距离**: {center_distance:.4f}")

                # 过滤掉未点亮、VF1不良的数据
                exclude_bins = ['未点亮', 'VF1不良']
                valid_mask = ~df[bin_col].isin(exclude_bins)
                valid_df = df[valid_mask].copy()  # 关键修改5：使用副本

                if len(valid_df) > 0:
                    # 计算原始集中度
                    points = valid_df[[ciex_col, ciey_col]].values
                    original_ratio = calculate_zone_ratio(points)

                    # 双滑块控制：集中度增益 + 中心点移动增益
                    col1, col2 = st.columns(2)

                    with col1:
                        concentration_gain = st.slider(
                            "集中度增益",
                            1.0, 5.0, 1.5, 0.1,
                            key="concentration_gain",
                            help="增益越大，色点分布越集中"
                        )

                    with col2:
                        center_shift_gain = st.slider(
                            "中心点移动增益",
                            0.0, 1.0, 0.0, 0.05,
                            key="center_shift_gain",
                            help="增益=0:保持原始中心; 增益=1:完全移动到目标中心(0.2771,0.26)"
                        )

                    # 计算实际中心点位置
                    center_vector = np.array(target_center) - np.array(original_center)
                    actual_center = np.array(original_center) + center_vector * center_shift_gain

                    st.write(f"**实际中心点位置**: ({actual_center[0]:.4f}, {actual_center[1]:.4f})")
                    st.write(f"**中心点移动程度**: {center_shift_gain * 100:.1f}% (0%:原始中心, 100%:目标中心)")

                    col3, col4 = st.columns(2)

                    with col3:
                        # 绘制原始色点
                        fig1, ax1 = plt.subplots(figsize=(8, 6))

                        colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_ZONES)))
                        for (zone_name, zone_points), color in zip(TARGET_ZONES.items(), colors):
                            polygon = Polygon(zone_points, closed=True, alpha=0.3,
                                              label=zone_name, color=color)
                            ax1.add_patch(polygon)

                        # 绘制原始中心点
                        ax1.plot(original_center[0], original_center[1], 'go', markersize=8,
                                 label='原始中心点', zorder=5)

                        # 绘制目标中心点
                        ax1.plot(target_center[0], target_center[1], 'ro', markersize=8,
                                 label='目标中心点', zorder=5)

                        ax1.scatter(valid_df[ciex_col], valid_df[ciey_col],
                                    s=20, alpha=0.6, c='blue', label=f'原始色点 (n={len(valid_df)})')

                        ax1.set_xlabel('CIE X 坐标', fontsize=10)
                        ax1.set_ylabel('CIE Y 坐标', fontsize=10)
                        ax1.set_title(f'原始色点分布 (目标色区占比: {original_ratio:.2%})', fontsize=12)
                        ax1.grid(True, alpha=0.3)

                        # 优化图例显示
                        legend1 = ax1.legend(
                            fontsize=7,
                            loc='upper right',
                            frameon=True,
                            framealpha=0.8,
                            borderpad=0.2,
                            handletextpad=0.1,
                            labelspacing=0.2,
                            borderaxespad=0.3
                        )
                        legend1.get_frame().set_edgecolor('gray')
                        legend1.get_frame().set_linewidth(0.5)

                        st.pyplot(fig1, use_container_width=True)

                    with col4:
                        # 应用增益并绘制调整后色点
                        adjusted_points, actual_center_calc = adjust_color_points_with_center_shift(
                            points.copy(),  # 关键修改6：传递points副本
                            original_center,
                            target_center,
                            concentration_gain,
                            center_shift_gain
                        )
                        adjusted_points = np.round(adjusted_points, 4)  # 标准化到4位小数
                        adjusted_ratio = calculate_zone_ratio(adjusted_points)

                        fig2, ax2 = plt.subplots(figsize=(8, 6))

                        for (zone_name, zone_points), color in zip(TARGET_ZONES.items(), colors):
                            polygon = Polygon(zone_points, closed=True, alpha=0.3,
                                              label=zone_name, color=color)
                            ax2.add_patch(polygon)

                        # 绘制原始中心点
                        ax2.plot(original_center[0], original_center[1], 'go', markersize=6,
                                 label='原始中心点', zorder=5, alpha=0.5)

                        # 绘制目标中心点
                        ax2.plot(target_center[0], target_center[1], 'ro', markersize=6,
                                 label='目标中心点', zorder=5, alpha=0.5)

                        # 绘制实际中心点
                        ax2.plot(actual_center[0], actual_center[1], 'mo', markersize=10,
                                 label='实际中心点', zorder=5)

                        ax2.scatter(adjusted_points[:, 0], adjusted_points[:, 1],
                                    s=20, alpha=0.6, c='red',
                                    label=f'调整后 (集中度增益={concentration_gain:.1f}, 中心移动={center_shift_gain:.2f})')

                        ax2.set_xlabel('CIE X 坐标', fontsize=10)
                        ax2.set_ylabel('CIE Y 坐标', fontsize=10)
                        ax2.set_title(f'调整后色点分布 (目标色区占比: {adjusted_ratio:.2%})', fontsize=12)
                        ax2.grid(True, alpha=0.3)

                        # 优化图例显示
                        legend2 = ax2.legend(
                            fontsize=7,
                            loc='upper right',
                            frameon=True,
                            framealpha=0.8,
                            borderpad=0.2,
                            handletextpad=0.1,
                            labelspacing=0.2,
                            borderaxespad=0.3
                        )
                        legend2.get_frame().set_edgecolor('gray')
                        legend2.get_frame().set_linewidth(0.5)

                        st.pyplot(fig2, use_container_width=True)

                    st.write(f"**优化效果**: 目标色区占比从 {original_ratio:.2%} 提升到 {adjusted_ratio:.2%}")
                    st.write(
                        f"**中心点位置**: 从原始中心 ({original_center[0]:.4f}, {original_center[1]:.4f}) 移动到 ({actual_center[0]:.4f}, {actual_center[1]:.4f})")

                    if st.button("确认应用色坐标调整", use_container_width=True):
                        # 应用色坐标调整（仅针对色坐标不良）
                        df, details, actual_center_final = apply_color_adjustment_with_center_shift(
                            df, original_center, target_center, concentration_gain, center_shift_gain, ciex_col,
                            ciey_col
                        )
                        st.session_state.current_df = df

                        if details:
                            st.success("✅ 色坐标不良数据已应用调整！")
                            st.write("**调整参数总结:**")
                            st.write(f"- 集中度增益: {concentration_gain:.1f}")
                            st.write(f"- 中心点移动增益: {center_shift_gain:.2f}")
                            st.write(f"- 实际中心点: ({actual_center_final[0]:.4f}, {actual_center_final[1]:.4f})")

                            # 重新计算调整后的占比
                            valid_mask_post = ~df[bin_col].isin(exclude_bins)
                            valid_mask_post &= (df[ciex_col] != 0) & (df[ciey_col] != 0) & (df[ciex_col] > 0.1) & (
                                    df[ciey_col] > 0.1)
                            valid_df_post = df[valid_mask_post].copy()  # 关键修改7：使用副本
                            points_post = valid_df_post[[ciex_col, ciey_col]].values
                            ratio_post = calculate_zone_ratio(points_post)
                            st.write(f"- 调整后目标色区占比: {ratio_post:.2%}")

                            st.write("**前5条修改详情:**")
                            for detail in details[:5]:
                                st.write(
                                    f"test_no {detail['test_no']}: ciex {detail['原ciex']}→{detail['新ciex']}, ciey {detail['原ciey']}→{detail['新ciey']}")

                            # ========== 核心修改：合并生成文件功能 ==========
                            # 生成包含所有修改的CSV文件（不良Bin+色坐标调整）
                            final_df = df.copy()

                            # 准备下载
                            output = io.BytesIO()
                            # 获取原始文件名
                            original_name = uploaded_file.name
                            if '.' in original_name:
                                original_base = original_name.rsplit('.', 1)[0]
                            else:
                                original_base = original_name
                            download_name = f"{original_base}_adjusted.csv"

                            # 保存为CSV（保持原格式）
                            final_df.to_csv(output, index=False, encoding='utf-8-sig')
                            output.seek(0)

                            # 生成下载链接
                            b64 = base64.b64encode(output.read()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="{download_name}">📥 下载包含所有修改的数据文件: {download_name}</a>'
                            st.markdown(href, unsafe_allow_html=True)

                            # 绘制最终的CIE图
                            st.subheader("📈 最终数据CIE坐标分布图")
                            plot_final_cie_chart(final_df, bin_col, ciex_col, ciey_col, TARGET_CENTER)
                        else:
                            st.info("没有找到色坐标不良数据")
            else:
                st.warning("无法计算原始色坐标中心，请检查数据格式")
        else:
            st.warning("缺少色坐标列 (ciex/ciey) 或 Bin代码列")

    else:
        st.info("请上传CSV数据文件开始分析")


if __name__ == "__main__":
    main()