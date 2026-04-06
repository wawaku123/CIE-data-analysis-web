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
import os

warnings.filterwarnings('ignore')

# ====================== 1. 页面与中文配置 ======================
st.set_page_config(
    layout="wide",
    page_title="CIE色点校正工具",
    page_icon="🔧"
)


# 修复matplotlib中文显示问题
def setup_chinese_font():
    """配置matplotlib中文显示，彻底解决方框问题"""
    try:
        font_list = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC', 'Heiti SC', 'WenQuanYi Micro Hei'
        ]
        for font_name in font_list:
            try:
                fm.findfont(fm.FontProperties(family=font_name))
                plt.rcParams['font.sans-serif'] = [font_name]
                break
            except:
                continue
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.warning(f"字体配置警告: {e}")
        plt.rcParams['font.sans-serif'] = plt.rcParamsDefault['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False


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
TARGET_CENTER = (0.2771, 0.26)  # 理想色点


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
    encodings_to_try = [encoding, 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin1']
    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            st.info(f"使用编码: {enc}")
            return df
        except UnicodeDecodeError:
            continue
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


def get_valid_mask(df, bin_col='bin_code', ciex_col='ciex', ciey_col='ciey'):
    """
    获取有效色点的过滤条件（核心：统一过滤逻辑）
    过滤规则：
    1. 排除未点亮、VF1不良
    2. 排除色坐标为0的点
    3. 只保留 0.2≤ciex≤0.4 且 0.2≤ciey≤0.4 的点
    """
    valid_mask = ~df[bin_col].isin(['未点亮', 'VF1不良'])
    valid_mask &= (df[ciex_col] >= 0.2) & (df[ciex_col] <= 0.4)
    valid_mask &= (df[ciey_col] >= 0.2) & (df[ciey_col] <= 0.4)
    valid_mask &= (df[ciex_col] != 0) & (df[ciey_col] != 0)
    return valid_mask


def calculate_original_color_center(df, bin_col='bin_code', ciex_col='ciex', ciey_col='ciey'):
    """计算原始色坐标中心（基于全部有效色点）"""
    valid_mask = get_valid_mask(df, bin_col, ciex_col, ciey_col)
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        st.warning("没有找到有效的色坐标数据（所有数据都被排除）")
        return None
    center_x = valid_df[ciex_col].mean()
    center_y = valid_df[ciey_col].mean()

    # 统计排除详情
    excluded_count = len(df) - len(valid_df)
    vf1_excluded = len(df[df[bin_col] == 'VF1不良'])
    unlit_excluded = len(df[df[bin_col] == '未点亮'])
    out_range_excluded = len(df[~((df[ciex_col] >= 0.2) & (df[ciex_col] <= 0.4) &
                                  (df[ciey_col] >= 0.2) & (df[ciey_col] <= 0.4))])
    zero_coord_excluded = len(df[(df[ciex_col] == 0) | (df[ciey_col] == 0)])

    st.write(f"**数据筛选详情**:")
    st.write(f"- 总数据点: {len(df)}")
    st.write(f"- 有效色点: {len(valid_df)} (用于中心点计算)")
    st.write(f"- 被排除的点: {excluded_count}")
    st.write(f"  - VF1不良: {vf1_excluded}")
    st.write(f"  - 未点亮: {unlit_excluded}")
    st.write(f"  - 超出0.2≤ciex≤0.4且0.2≤ciey≤0.4范围: {out_range_excluded}")
    st.write(f"  - 色坐标为0: {zero_coord_excluded}")

    return (center_x, center_y)


def adjust_color_points_for_all_valid(
        df, original_center, target_center,
        concentration_gain=1.0, custom_shift_x=None, custom_shift_y=None,
        bin_col='bin_code', ciex_col='ciex', ciey_col='ciey'
):
    """
    核心修复：对全部有效色点进行调整（而非仅色坐标不良）
    1. 统一过滤有效色点
    2. 对所有有效点应用移动+集中度调整
    3. 移动值保留四位小数，确保位数一致
    """
    # 1. 获取有效色点掩码
    valid_mask = get_valid_mask(df, bin_col, ciex_col, ciey_col)
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        st.warning("无有效色点可调整")
        return df, [], original_center, 0, 0

    # 2. 计算移动值（优先使用自定义值，否则用原始-理想差值）
    default_shift_x = target_center[0] - original_center[0]
    default_shift_y = target_center[1] - original_center[1]
    shift_x = custom_shift_x if custom_shift_x is not None else default_shift_x
    shift_y = custom_shift_y if custom_shift_y is not None else default_shift_y

    # 3. 保留四位小数（关键：统一精度）
    shift_x = round(shift_x, 4)
    shift_y = round(shift_y, 4)
    actual_center = (
        round(original_center[0] + shift_x, 4),
        round(original_center[1] + shift_y, 4)
    )

    # 4. 提取有效点坐标
    original_points = valid_df[[ciex_col, ciey_col]].values

    # 5. 应用调整逻辑（和预览图完全一致）
    # 步骤1：平移到原始中心
    points_centered = original_points - original_center
    # 步骤2：应用集中度缩放
    scale_factor = 1.0 / np.sqrt(concentration_gain) if concentration_gain > 0 else 1.0
    points_scaled = points_centered * scale_factor
    # 步骤3：移动到实际中心 + 保留四位小数
    adjusted_points = points_scaled + actual_center
    adjusted_points = np.round(adjusted_points, 4)  # 强制四位小数

    # 6. 将调整后的值写回原数据
    df_copy = df.copy()
    df_copy.loc[valid_mask, ciex_col] = adjusted_points[:, 0]
    df_copy.loc[valid_mask, ciey_col] = adjusted_points[:, 1]

    # 7. 记录修改详情（前10条）
    modification_details = []
    valid_indices = df_copy[valid_mask].index[:10]  # 只记录前10条
    for idx in valid_indices:
        test_no_col = 'test_no' if 'test_no' in df_copy.columns else 'TestNo' if 'TestNo' in df_copy.columns else None
        test_no = df_copy.at[idx, test_no_col] if test_no_col else idx
        modification_details.append({
            'test_no': test_no,
            '原ciex': round(df.at[idx, ciex_col], 4),
            '新ciex': df_copy.at[idx, ciex_col],
            '原ciey': round(df.at[idx, ciey_col], 4),
            '新ciey': df_copy.at[idx, ciey_col]
        })

    return df_copy, modification_details, actual_center, shift_x, shift_y


def get_normal_dvf_range(df):
    """计算正常材料的DVF范围（排除不良）"""
    if 'dvf' not in df.columns or 'bin_code' not in df.columns:
        return -0.043, -0.045
    normal_mask = ~df['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_df = df[normal_mask]
    if len(normal_df) == 0 or len(normal_df['dvf'].dropna()) == 0:
        return -0.043, -0.045
    dvf_values = normal_df['dvf'].dropna()
    return dvf_values.quantile(0.25), dvf_values.quantile(0.75)


def modify_dvf_smart(df, modify_ratio=1.0):
    """智能修改DVF不良数据"""
    if 'dvf' not in df.columns or 'bin_code' not in df.columns:
        return df, 0, []
    mask = (df['bin_code'] == 'DVF不良')
    total_bad = mask.sum()
    if total_bad == 0:
        return df, 0, []
    modify_count = int(total_bad * modify_ratio)
    dvf_min, dvf_max = get_normal_dvf_range(df)
    normal_mask = ~df['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_dvf_values = df[normal_mask]['dvf'].dropna().values
    if len(normal_dvf_values) > 0:
        if len(normal_dvf_values) >= modify_count:
            new_values = np.random.choice(normal_dvf_values, size=modify_count, replace=True)
        else:
            new_values = np.random.uniform(dvf_min, dvf_max, size=modify_count)
    else:
        new_values = np.random.uniform(dvf_min, dvf_max, size=modify_count)
    bad_indices = df[mask].index.tolist()
    modify_indices = bad_indices[:modify_count]
    modification_details = []
    for i, idx in enumerate(modify_indices):
        original_value = df.at[idx, 'dvf']
        original_decimal = len(str(original_value).split('.')[1]) if '.' in str(original_value) else 3
        new_value = new_values[i] if i < len(new_values) else np.random.uniform(dvf_min, dvf_max)
        new_value = round(new_value, original_decimal)
        df.at[idx, 'dvf'] = new_value
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
    """智能修改VF2不良数据"""
    vf2_cols = ['forward_voltage2_V', 'forward_voltage2', 'forward_vc', 'forward_vcf']
    vf2_col = None
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
    modify_count = int(total_bad * modify_ratio)
    normal_mask = ~df['bin_code'].str.contains('不良|未点亮|已修正', na=False)
    normal_vf2_values = df[normal_mask][vf2_col].dropna().values
    if len(normal_vf2_values) > 0:
        vf2_min = np.percentile(normal_vf2_values, 25)
        vf2_max = np.percentile(normal_vf2_values, 75)
        if len(normal_vf2_values) >= modify_count:
            new_values = np.random.choice(normal_vf2_values, size=modify_count, replace=True)
        else:
            new_values = np.random.uniform(vf2_min, vf2_max, size=modify_count)
    else:
        vf2_min, vf2_max = 4.7, 5.0
        new_values = np.random.uniform(vf2_min, vf2_max, size=modify_count)
    bad_indices = df[mask].index.tolist()
    modify_indices = bad_indices[:modify_count]
    modification_details = []
    for i, idx in enumerate(modify_indices):
        original_value = df.at[idx, vf2_col]
        original_decimal = len(str(original_value).split('.')[1]) if '.' in str(original_value) else 3
        new_value = new_values[i] if i < len(new_values) else np.random.uniform(vf2_min, vf2_max)
        new_value = round(new_value, original_decimal)
        df.at[idx, vf2_col] = new_value
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


def plot_cie_chart(df, bin_col, ciex_col, ciey_col, title, center_points=None, ratio=0.0, figsize=(8, 6)):
    """统一绘制CIE图表（确保预览/最终图样式一致）"""
    valid_mask = get_valid_mask(df, bin_col, ciex_col, ciey_col)
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        st.warning("无有效色坐标数据")
        return

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_ZONES)))

    # 绘制目标色区
    for (zone_name, zone_points), color in zip(TARGET_ZONES.items(), colors):
        polygon = Polygon(zone_points, closed=True, alpha=0.3, label=zone_name, color=color)
        ax.add_patch(polygon)

    # 绘制中心点
    if center_points:
        for center in center_points:
            ax.plot(center['x'], center['y'], center['marker'],
                    markersize=center['size'], label=center['label'], zorder=5, alpha=center.get('alpha', 1.0))

    # 绘制色点
    ax.scatter(valid_df[ciex_col], valid_df[ciey_col],
               s=20, alpha=0.6, c=center_points[0]['color'] if center_points else 'darkred',
               label=f'色点 (n={len(valid_df)})')

    # 图表配置
    ax.set_xlabel('CIE X 坐标', fontsize=12)
    ax.set_ylabel('CIE Y 坐标', fontsize=12)
    ax.set_title(f'{title}（目标色区占比: {ratio:.2%}）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.8, borderpad=0.3)

    st.pyplot(fig, use_container_width=False)


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
    st.markdown("上传LED分光数据文件，**对全部有效色点**进行坐标调整，优化集中性")

    # 清除缓存按钮
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🗑️ 清除缓存数据", use_container_width=True, type='secondary'):
            clear_cache_data()

    # 文件上传
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

    if uploaded_file is not None:
        # 初始化原始数据和当前数据
        if 'original_df' not in st.session_state:
            st.session_state.original_df = read_csv_file(uploaded_file)
        if 'current_df' not in st.session_state:
            st.session_state.current_df = st.session_state.original_df.copy()

        df = st.session_state.current_df
        original_df = st.session_state.original_df

        # 识别关键列
        key_columns_map = {
            '测试编号': ['test_no', 'TestNo', 'Test_No'],
            'Bin代码': ['bin_code', 'BinCode', 'Bin_Code'],
            'DVF': ['dvf', 'DVF', 'ΔVF'],
            'VF2': ['forward_voltage2_V', 'forward_voltage2', 'forward_vc', 'forward_vcf'],
            'CIE x': ['ciex', 'CIE_x', 'x_coordinate'],
            'CIE y': ['ciey', 'CIE_y', 'y_coordinate']
        }
        key_columns_found = {}
        for col_name, possible_names in key_columns_map.items():
            for name in possible_names:
                if name in df.columns:
                    key_columns_found[col_name] = name
                    break

        # 显示列名信息
        st.subheader("📊 数据列名信息")
        st.write(f"总列数: {len(df.columns)}")
        st.write("**关键列识别:**")
        for col_name, found_col in key_columns_found.items():
            st.write(f"✓ {col_name}: {found_col}")

        # 原始数据统计
        st.subheader("原始数据分析（仅上传文件数据）")
        st.write(f"总数据点数: {len(original_df)}")
        bin_col = key_columns_found.get('Bin代码')
        if bin_col:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("不同Bin数量", len(original_df[bin_col].unique()))
            with col2:
                st.metric("总行数", len(original_df))
            with col3:
                bad_bins = ['未点亮', '色坐标不良', 'DVF不良', 'VF2不良']
                st.metric("不良品数量", len(original_df[original_df[bin_col].isin(bad_bins)]))

        # 1. Bin统计
        if bin_col:
            st.subheader("1. Bin数量统计（原始数据）")
            bin_stats = original_df[bin_col].value_counts().reset_index()
            bin_stats.columns = ['Bin Code', '数量']
            bin_stats['占比'] = (bin_stats['数量'] / len(original_df) * 100).round(2).astype(str) + '%'
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(bin_stats, height=300, use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                top_bins = bin_stats.head(10)
                ax.barh(top_bins['Bin Code'][::-1], top_bins['数量'][::-1])
                ax.set_xlabel('数量', fontsize=10)
                ax.set_title('Top 10 Bin分布（原始数据）', fontsize=12)
                st.pyplot(fig, use_container_width=True)

        # 2. 不良Bin校正（VF2/DVF）
        st.subheader("2. 不良Bin校正（VF2/DVF）")
        if bin_col:
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

        # 3. 色坐标全量调整（核心功能）
        st.subheader("3. 色坐标全量优化（对所有有效色点调整）")
        ciex_col = key_columns_found.get('CIE x')
        ciey_col = key_columns_found.get('CIE y')
        if ciex_col and ciey_col and bin_col:
            # 计算原始中心点（基于全部有效点）
            original_center = calculate_original_color_center(original_df, bin_col, ciex_col, ciey_col)
            target_center = TARGET_CENTER

            if original_center is not None:
                st.write(f"**原始色坐标中心**: ({original_center[0]:.6f}, {original_center[1]:.6f})")
                st.write(f"**理想色坐标中心**: ({target_center[0]:.6f}, {target_center[1]:.6f})")

                # 计算默认移动值
                default_shift_x = target_center[0] - original_center[0]
                default_shift_y = target_center[1] - original_center[1]
                st.write(f"**默认移动值（实际-理想）**:")
                st.write(f"- ciex 移动值: {default_shift_x:.4f}（四位小数）")
                st.write(f"- ciey 移动值: {default_shift_y:.4f}（四位小数）")

                # 获取原始有效点
                valid_mask_original = get_valid_mask(original_df, bin_col, ciex_col, ciey_col)
                valid_df_original = original_df[valid_mask_original].copy()
                original_points = valid_df_original[[ciex_col, ciey_col]].values
                original_ratio = calculate_zone_ratio(original_points)

                # 调整参数设置
                st.write("### 调整参数设置")
                col1, col2 = st.columns(2)
                with col1:
                    concentration_gain = st.slider(
                        "集中度增益", 1.0, 5.0, 1.5, 0.1, key="concentration_gain",
                        help="增益越大，色点分布越集中（1.0=不缩放，5.0=最集中）"
                    )
                    custom_shift_x = st.number_input(
                        "ciex 移动值（四位小数）",
                        value=round(default_shift_x, 4),
                        step=0.0001, format="%.4f", key="shift_x",
                        help=f"默认值：{round(default_shift_x, 4)}（原始与理想的差值）"
                    )
                with col2:
                    custom_shift_y = st.number_input(
                        "ciey 移动值（四位小数）",
                        value=round(default_shift_y, 4),
                        step=0.0001, format="%.4f", key="shift_y",
                        help=f"默认值：{round(default_shift_y, 4)}（原始与理想的差值）"
                    )

                # 预览调整效果
                st.write("### 调整效果预览")
                col3, col4 = st.columns(2)
                with col3:
                    # 原始图
                    center_points_original = [{
                        'x': original_center[0], 'y': original_center[1],
                        'marker': 'go', 'size': 8, 'label': '原始中心点', 'color': 'blue'
                    }, {
                        'x': target_center[0], 'y': target_center[1],
                        'marker': 'ro', 'size': 8, 'label': '理想中心点', 'color': 'blue'
                    }]
                    plot_cie_chart(
                        original_df, bin_col, ciex_col, ciey_col,
                        title="原始色点分布", center_points=center_points_original,
                        ratio=original_ratio, figsize=(8, 6)
                    )
                with col4:
                    # 预览调整后图（和实际修改逻辑完全一致）
                    temp_points = original_points.copy()
                    shift_x_preview = round(custom_shift_x, 4)
                    shift_y_preview = round(custom_shift_y, 4)
                    actual_center_preview = (
                        round(original_center[0] + shift_x_preview, 4),
                        round(original_center[1] + shift_y_preview, 4)
                    )
                    # 应用调整
                    points_centered = temp_points - original_center
                    scale_factor = 1.0 / np.sqrt(concentration_gain)
                    points_scaled = points_centered * scale_factor
                    adjusted_points_preview = np.round(points_scaled + actual_center_preview, 4)
                    adjusted_ratio_preview = calculate_zone_ratio(adjusted_points_preview)

                    # 绘制预览图
                    center_points_adjusted = [{
                        'x': original_center[0], 'y': original_center[1],
                        'marker': 'go', 'size': 6, 'label': '原始中心点', 'alpha': 0.5, 'color': 'red'
                    }, {
                        'x': target_center[0], 'y': target_center[1],
                        'marker': 'ro', 'size': 6, 'label': '理想中心点', 'alpha': 0.5, 'color': 'red'
                    }, {
                        'x': actual_center_preview[0], 'y': actual_center_preview[1],
                        'marker': 'mo', 'size': 10, 'label': '实际中心点', 'color': 'red'
                    }]
                    # 临时构建预览数据
                    preview_df = original_df.copy()
                    preview_df.loc[valid_mask_original, ciex_col] = adjusted_points_preview[:, 0]
                    preview_df.loc[valid_mask_original, ciey_col] = adjusted_points_preview[:, 1]
                    plot_cie_chart(
                        preview_df, bin_col, ciex_col, ciey_col,
                        title="调整后色点分布", center_points=center_points_adjusted,
                        ratio=adjusted_ratio_preview, figsize=(8, 6)
                    )

                # 显示优化效果
                st.write(f"**优化效果预览**: 目标色区占比从 {original_ratio:.2%} 提升到 {adjusted_ratio_preview:.2%}")
                st.write(
                    f"**中心点移动预览**: 从原始中心 ({original_center[0]:.4f}, {original_center[1]:.4f}) 移动到 ({actual_center_preview[0]:.4f}, {actual_center_preview[1]:.4f})")

                # 确认应用调整
                if st.button("确认应用色坐标全量调整", use_container_width=True, type='primary'):
                    # 对全部有效点进行调整
                    df_adjusted, details, actual_center_final, shift_x_used, shift_y_used = adjust_color_points_for_all_valid(
                        df, original_center, target_center, concentration_gain,
                        custom_shift_x, custom_shift_y, bin_col, ciex_col, ciey_col
                    )
                    st.session_state.current_df = df_adjusted

                    # 计算最终占比
                    valid_mask_final = get_valid_mask(df_adjusted, bin_col, ciex_col, ciey_col)
                    valid_df_final = df_adjusted[valid_mask_final].copy()
                    final_points = valid_df_final[[ciex_col, ciey_col]].values
                    final_ratio = calculate_zone_ratio(final_points)

                    # 显示调整结果
                    st.success("✅ 已对**全部有效色点**应用色坐标调整！")
                    st.write("**调整参数总结:**")
                    st.write(f"- 集中度增益: {concentration_gain:.1f}")
                    st.write(f"- 实际使用的移动值（四位小数）:")
                    st.write(f"  - ciex: {shift_x_used:.4f}")
                    st.write(f"  - ciey: {shift_y_used:.4f}")
                    st.write(f"- 最终中心点: ({actual_center_final[0]:.4f}, {actual_center_final[1]:.4f})")
                    st.write(f"- 最终目标色区占比: {final_ratio:.2%}")

                    # 显示修改详情
                    st.write("**前10条修改详情（全部有效点）:**")
                    for detail in details:
                        st.write(
                            f"test_no {detail['test_no']}: "
                            f"ciex {detail['原ciex']:.4f}→{detail['新ciex']:.4f}, "
                            f"ciey {detail['原ciey']:.4f}→{detail['新ciey']:.4f}"
                        )

                    # 生成下载文件
                    output = io.BytesIO()
                    original_name = uploaded_file.name
                    original_base = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
                    download_name = f"{original_base}_adjusted.csv"
                    df_adjusted.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)
                    b64 = base64.b64encode(output.read()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{download_name}">📥 下载调整后数据文件: {download_name}</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # 绘制最终图（和预览图完全一致）
                    st.subheader("📈 最终色点分布（全量调整后）")
                    center_points_final = [{
                        'x': actual_center_final[0], 'y': actual_center_final[1],
                        'marker': 'mo', 'size': 10, 'label': '最终中心点', 'color': 'darkred'
                    }, {
                        'x': target_center[0], 'y': target_center[1],
                        'marker': 'ro', 'size': 8, 'label': '理想中心点 (0.2771, 0.26)', 'color': 'darkred'
                    }]
                    plot_cie_chart(
                        df_adjusted, bin_col, ciex_col, ciey_col,
                        title="最终色点分布", center_points=center_points_final,
                        ratio=final_ratio, figsize=(8, 6)
                    )
        else:
            st.warning("缺少关键列：请确保数据包含 Bin代码、CIE x、CIE y 列")
    else:
        st.info("请上传CSV数据文件开始分析")


if __name__ == "__main__":
    main()
