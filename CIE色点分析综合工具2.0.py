import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import time



# 提高图表分辨率
st.set_page_config(
    page_title="CIE色点分析综合工具",
    page_icon="🌈",
    layout="wide"
)

# 预定义色区数据（固定不变）
COLOR_ZONE_PRESETS = {
    "NCSP": {
        "name": "NCSP色区 (成品)",
        "zones": {
            'DK32': [(0.2743, 0.265), (0.277, 0.27), (0.2825, 0.27), (0.2798, 0.265)],
            'DK33': [(0.2715, 0.26), (0.2743, 0.265), (0.2798, 0.265), (0.2771, 0.26)],
            'DK34': [(0.2688, 0.255), (0.2715, 0.26), (0.2771, 0.26), (0.2743, 0.255)],
            'DK35': [(0.2661, 0.25), (0.2688, 0.255), (0.2743, 0.255), (0.2716, 0.25)],
            'DL32': [(0.2798, 0.265), (0.2825, 0.27), (0.288, 0.27), (0.2853, 0.265)],
            'DL33': [(0.2771, 0.26), (0.2798, 0.265), (0.2853, 0.265), (0.2826, 0.26)],
            'DL34': [(0.2743, 0.255), (0.2771, 0.26), (0.2826, 0.26), (0.2799, 0.255)],
            'DL35': [(0.2716, 0.25), (0.2743, 0.255), (0.2799, 0.255), (0.2771, 0.25)],
            'DM32': [(0.2853, 0.265), (0.288, 0.27), (0.2936, 0.27), (0.2908, 0.265)],
            'DM33': [(0.2826, 0.26), (0.2853, 0.265), (0.2908, 0.265), (0.2881, 0.26)],
            'DM34': [(0.2799, 0.255), (0.2826, 0.26), (0.2881, 0.26), (0.2854, 0.255)],
            'DM35': [(0.2771, 0.25), (0.2799, 0.255), (0.2854, 0.255), (0.2827, 0.25)]
        }
    },
    "CSP": {
        "name": "CSP色区",
        "sub_types": {
            "M": {
                "name": "一次模压",
                "zones": {
                    'DK32_M': [(0.2999, 0.3401), (0.3026, 0.3451), (0.3081, 0.3451), (0.3054, 0.3401)],
                    'DK33_M': [(0.2971, 0.3351), (0.2999, 0.3401), (0.3054, 0.3401), (0.3027, 0.3351)],
                    'DK34_M': [(0.2944, 0.3301), (0.2971, 0.3351), (0.3027, 0.3351), (0.2999, 0.3301)],
                    'DK35_M': [(0.2917, 0.3251), (0.2944, 0.3301), (0.2999, 0.3301), (0.2972, 0.3251)],
                    'DL32_M': [(0.3054, 0.3401), (0.3081, 0.3451), (0.3136, 0.3451), (0.3109, 0.3401)],
                    'DL33_M': [(0.3027, 0.3351), (0.3054, 0.3401), (0.3109, 0.3401), (0.3082, 0.3351)],
                    'DL34_M': [(0.2999, 0.3301), (0.3027, 0.3351), (0.3082, 0.3351), (0.3055, 0.3301)],
                    'DL35_M': [(0.2972, 0.3251), (0.2999, 0.3301), (0.3055, 0.3301), (0.3027, 0.3251)]
                }
            },
            "C": {
                "name": "无水切割",
                "zones": {
                    'DK32_C': [(0.3117, 0.3540), (0.3144, 0.3590), (0.3199, 0.3590), (0.3172, 0.3540)],
                    'DK33_C': [(0.3089, 0.3490), (0.3117, 0.3540), (0.3172, 0.3540), (0.3145, 0.3490)],
                    'DK34_C': [(0.3062, 0.3440), (0.3089, 0.3490), (0.3145, 0.3490), (0.3117, 0.3440)],
                    'DK35_C': [(0.3035, 0.3390), (0.3062, 0.3440), (0.3117, 0.3440), (0.3090, 0.3390)],
                    'DL32_C': [(0.3172, 0.3540), (0.3199, 0.3590), (0.3254, 0.3590), (0.3227, 0.3540)],
                    'DL33_C': [(0.3145, 0.3490), (0.3172, 0.3540), (0.3227, 0.3540), (0.3200, 0.3490)],
                    'DL34_C': [(0.3117, 0.3440), (0.3145, 0.3490), (0.3200, 0.3490), (0.3173, 0.3440)],
                    'DL35_C': [(0.3090, 0.3390), (0.3117, 0.3440), (0.3173, 0.3440), (0.3145, 0.3390)]
                }
            }
        }
    }
}

# 定义产出分布统计的Bin区范围
PRODUCTION_BINS = {
    "Wavelength": {
        "column": "peak_wavelength1_nm",
        "name": "峰值波长",
        "units": "nm",
        "bins": {
            "H": (446, 448),
            "J": (448, 450),
            "K": (450, 452),
            "L": (452, 454),
            "M": (454, 456),
            "N": (456, 458),
            "P": (458, 460)
        },
        "order": ["H", "J", "K", "L", "M", "N", "P"]
    },
    "Brightness": {
        "column": "LuminousFlux_lm",
        "name": "亮度",
        "units": "lm",
        "bins": {
            "CP": (2.8, 3.04),
            "CQ": (3.04, 3.30),
            "CR": (3.3, 3.59),
            "CS": (3.59, 3.9),
            "CT": (3.9, 4.19)
        },
        "order": ["CP", "CQ", "CR", "CS", "CT"]
    },
    "Voltage": {
        "column": "forward_voltage1_V",
        "name": "电压",
        "units": "V",
        "bins": {
            "N4": (5.4, 5.6),
            "S4": (5.6, 5.8),
            "W4": (5.8, 6.0)
        },
        "order": ["N4", "S4", "W4"]
    }
}

# 自定义颜色映射
color_list = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# 预定义的色阶方案
color_scales = {
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Magma': 'magma',
    'Cividis': 'cividis',
    'RdBu': 'RdBu',
    'Jet': 'jet',
    'Rainbow': 'rainbow'
}


# 判断点是否在多边形内部的函数
def point_in_polygon(point, polygon):
    """
    判断点是否在多边形内部
    point: (x, y) 元组
    polygon: 多边形顶点列表，每个顶点是(x, y)元组
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # 检查点是否在多边形的边上
        if ((yi > y) != (yj > y)):
            x_intersect = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x <= x_intersect:
                inside = not inside

    return inside


# 将数值映射到对应的Bin区
def value_to_bin(value, bins):
    """将数值映射到对应的Bin区"""
    # 处理NaN值
    if pd.isna(value):
        return "NaN"

    for bin_code, (min_val, max_val) in bins.items():
        if min_val <= value < max_val:
            return bin_code
    return "Out of Range"


# 缓存数据加载函数
@st.cache_data(show_spinner=False)
def load_data(file, product_type, encoding='gbk'):
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(file)
    elif file_ext == '.csv':
        try:
            df = pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            st.error(f"无法使用 {encoding} 编码读取 CSV 文件")
            return None
    else:
        st.error(f"不支持的文件格式: {file_ext}")
        return None

    # 检查必要的列是否存在
    if product_type == "CSP":
        required_columns = ['pos_x', 'pos_y', 'ciex', 'ciey', 'bin_code',
                            'peak_wavelength1_nm', 'LuminousFlux_lm', 'forward_voltage1_V']
    else:  # NCSP产品
        required_columns = ['PosX_Map', 'PosY_Map', 'ciex', 'ciey', 'bin_code',
                            'peak_wavelength1_nm', 'LuminousFlux_lm', 'forward_voltage1_V']

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"文件缺少必要的列: {', '.join(missing_columns)}")
        return None

    # 坐标修正：仅对NCSP产品进行修正
    if product_type == "NCSP":
        df.loc[df['PosX_Map'] == 68, 'PosX_Map'] = 78

    return df


# 颜色转换函数：将十六进制颜色转换为RGBA格式
def hex_to_rgba(hex_color, alpha=0.2):
    """将十六进制颜色转换为RGBA字符串"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    else:
        return hex_color  # 如果格式不正确，返回原始值


# --------------------------
# 新增：1. 坐标轴固定比例计算函数
# --------------------------
def calculate_fixed_ratio_range(all_ciex, all_ciey):
    """
    计算固定比例的坐标轴范围：
    - x轴：9个刻度，每个0.0055，总范围0.044
    - y轴：6个刻度，每个0.01，总范围0.05
    - 基于数据中心点居中显示
    """
    if not all_ciex or not all_ciey:
        return [0, 0.044], [0, 0.05]  # 默认范围

    # 计算数据中心点
    x_center = np.mean(all_ciex)
    y_center = np.mean(all_ciey)

    # 固定总范围（刻度数-1 * 刻度单位）
    x_total_range = 0.0055 * (9 - 1)  # 9个刻度 → 8个间隔
    y_total_range = 0.01 * (6 - 1)  # 6个刻度 → 5个间隔

    # 计算边界（居中显示）
    x_min = x_center - x_total_range / 2
    x_max = x_center + x_total_range / 2
    y_min = y_center - y_total_range / 2
    y_max = y_center + y_total_range / 2

    return [x_min, x_max], [y_min, y_max]


# --------------------------
# 新增：2. 斜率计算相关函数
# --------------------------
def find_zone_containing_point(target_point, color_zones):
    """找到包含目标点的色区（平行四边形）"""
    for zone_name, zone_coords in color_zones.items():
        if point_in_polygon(target_point, zone_coords):
            return zone_name, zone_coords
    return None, None


def calculate_parallelogram_positive_slopes(zone_coords):
    """计算平行四边形的正斜率（仅返回>0的斜率，去重）"""
    if len(zone_coords) != 4:
        return None

    # 计算四条边的斜率
    slopes = []
    for i in range(4):
        p1 = zone_coords[i]
        p2 = zone_coords[(i + 1) % 4]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else float('inf')
        slopes.append(round(slope, 6))  # 保留6位小数去重

    # 筛选正斜率，去重（平行四边形对边斜率相等）
    positive_slopes = list({slope for slope in slopes if slope > 0 and slope != float('inf')})
    return positive_slopes[0] if positive_slopes else None  # 返回唯一正斜率


def get_slope_line_params(target_point, slope):
    """根据目标点和斜率计算直线方程（y = mx + b），生成直线点"""
    x0, y0 = target_point
    b = y0 - slope * x0  # 截距
    equation = f"y = {slope:.4f}x + {b:.6f}"
    # 生成直线的x范围（基于目标点±0.02，确保覆盖色区）
    x_line = [x0 - 0.02, x0 + 0.02]
    y_line = [slope * x + b for x in x_line]
    return equation, (x_line, y_line)

# 生成带色区的交互式CIE散点图（支持中心点移动，色区位置固定）

@st.cache_data(show_spinner=False)
def generate_interactive_cie_plot_with_zones(df_dict, selected_bin_codes, colors, title, fig_width, fig_height,
                                             point_size, alpha, x_label, y_label, show_grid, x_range, y_range,
                                             selected_zones=None, color_zones=None, move_center=False,
                                             target_center=(0.2771, 0.26),
                                             # 新增：斜率直线参数
                                             show_slope_line=False, slope_line_info=None):


    # 创建基础图形
    fig = go.Figure()

    # 计算补偿系数（仅当需要移动中心点时）
    offset_x, offset_y = 0, 0
    if move_center and df_dict and selected_bin_codes:
        # 计算所有选中数据的平均中心点
        all_ciex = []
        all_ciey = []
        for df in df_dict.values():
            filtered_df = df[df['bin_code'].isin(selected_bin_codes)]
            if not filtered_df.empty:
                all_ciex.extend(filtered_df['ciex'].tolist())
                all_ciey.extend(filtered_df['ciey'].tolist())

        if all_ciex and all_ciey:
            actual_center_x = np.mean(all_ciex)
            actual_center_y = np.mean(all_ciey)
            # 计算补偿系数
            offset_x = target_center[0] - actual_center_x
            offset_y = target_center[1] - actual_center_y

    # 为每个数据源添加散点（应用移动补偿）
    for file_name, df in df_dict.items():
        # 筛选bin_code
        filtered_df = df[df['bin_code'].isin(selected_bin_codes)]

        if not filtered_df.empty:
            color = colors.get(file_name, '#1f77b4')

            # 如果需要移动中心点，则应用补偿（只移动数据点，不移动色区）
            x_values = filtered_df['ciex'] + (offset_x if move_center else 0)
            y_values = filtered_df['ciey'] + (offset_y if move_center else 0)

            # 添加散点
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=color,
                        opacity=alpha,
                        line=dict(width=1, color='black')
                    ),
                    name=file_name,
                    customdata=list(zip(
                        filtered_df.get('PosX_Map', filtered_df.get('pos_x')),
                        filtered_df.get('PosY_Map', filtered_df.get('pos_y')),
                        filtered_df['ciex'],
                        filtered_df['ciey'],
                        filtered_df['bin_code']
                    )),
                    hovertemplate=(
                        f"PosX: %{{customdata[0]}}<br>"
                        f"PosY: %{{customdata[1]}}<br>"
                        f"原始ciex: %{{customdata[2]:.4f}}<br>"
                        f"原始ciey: %{{customdata[3]:.4f}}<br>"
                        f"移动后ciex: %{{x:.4f}}<br>"
                        f"移动后ciey: %{{y:.4f}}<br>"
                        f"bin_code: %{{customdata[4]}}<extra></extra>"
                    )
                )
            )

    # 添加色区显示（色区位置固定，不随中心点移动）
    if selected_zones is None:
        selected_zones = list(color_zones.keys())

    zone_colors = {zone: color_list[i % len(color_list)] for i, zone in enumerate(color_zones.keys())}

    for zone_name, zone_coords in color_zones.items():
        if zone_name in selected_zones:
            # 色区坐标不应用偏移，保持固定
            x_coords = [p[0] for p in zone_coords] + [zone_coords[0][0]]  # 闭合多边形
            y_coords = [p[1] for p in zone_coords] + [zone_coords[0][1]]

            # 添加填充区域（半透明）
            rgba_color = hex_to_rgba(zone_colors[zone_name], 0.2)  # 使用0.2的透明度
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    fill='toself',
                    fillcolor=rgba_color,
                    line=dict(
                        color=zone_colors[zone_name],
                        width=2
                    ),
                    name=zone_name,
                    hoverinfo='name',
                    showlegend=True
                )
            )

    # 添加中心点标记和统计信息
    stats_data = []
    for file_name, df in df_dict.items():
        filtered_df = df[df['bin_code'].isin(selected_bin_codes)]
        if not filtered_df.empty:
            # 计算原始中心点
            original_center_x = filtered_df['ciex'].mean()
            original_center_y = filtered_df['ciey'].mean()

            # 计算移动后的中心点
            center_x = original_center_x + (offset_x if move_center else 0)
            center_y = original_center_y + (offset_y if move_center else 0)

            # 计算标准差
            std_x = filtered_df['ciex'].std()
            std_y = filtered_df['ciey'].std()

            # 计算颜色一致性指标
            distances = np.sqrt(
                (filtered_df['ciex'] - original_center_x) ** 2 + (filtered_df['ciey'] - original_center_y) ** 2)
            color_consistency = distances.std()

            # 添加中心点标记
            fig.add_trace(
                go.Scatter(
                    x=[center_x],
                    y=[center_y],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=colors.get(file_name, '#1f77b4'),
                        symbol='x',
                        line=dict(width=2, color='black')
                    ),
                    text=[file_name],
                    textposition='top center',
                    showlegend=False,
                    hovertemplate=(
                        f"原始中心点: ({original_center_x:.4f}, {original_center_y:.4f})<br>"
                        f"移动后中心点: ({center_x:.4f}, {center_y:.4f})<extra></extra>"
                    )
                )
            )

            # 保存统计数据
            stats_data.append({
                '材料': file_name,
                '原始中心点 x': original_center_x,
                '原始中心点 y': original_center_y,
                '移动后中心点 x': center_x if move_center else None,
                '移动后中心点 y': center_y if move_center else None,
                'x 标准差': std_x,
                'y 标准差': std_y,
                '颜色一致性': color_consistency,
                '样本数': len(filtered_df)
            })

    # 如果启用了中心点移动，添加目标中心点标记
    if move_center:
        fig.add_trace(
            go.Scatter(
                x=[target_center[0]],
                y=[target_center[1]],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='purple',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                text=["目标中心"],
                textposition='bottom center',
                showlegend=False,
                hovertemplate=f"目标中心点: ({target_center[0]:.4f}, {target_center[1]:.4f})<extra></extra>"
            )
        )
    # --------------------------
    # 新增：绘制斜率分析直线
    # --------------------------
    if show_slope_line and slope_line_info:
        equation, (x_line, y_line) = slope_line_info
        # 添加直线
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='darkred', width=2.5, dash='solid'),
            name=f'理想斜率线: {equation}',
            hoverinfo='name'
        ))
        # 添加方程标注（固定在图左下角，避免遮挡）
        fig.add_annotation(
            x=x_range[0] + (x_range[1] - x_range[0]) * 0.02,
            y=y_range[0] + (y_range[1] - y_range[0]) * 0.02,
            text=f'理想直线方程: {equation}',
            showarrow=False,
            font=dict(size=11, color='darkred', weight='bold'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='darkred',
            borderwidth=1
        )

    """生成带色区显示的交互式CIE散点图，支持中心点移动功能，色区位置保持固定"""
    if color_zones is None:
        color_zones = COLOR_ZONE_PRESETS["NCSP"]["zones"]  # 默认使用NCSP色区

    # 设置图表布局
    fig.update_layout(
        title=title + (" (已应用中心点移动)" if move_center else ""),
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=fig_width,
        height=fig_height,
        hovermode='closest',
        dragmode='zoom',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        transition_duration=200,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="SimHei"
        )
    )

    # 设置坐标轴范围
    if x_range:
        fig.update_xaxes(range=x_range)
    if y_range:
        fig.update_yaxes(range=y_range)

    # 添加网格线
    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig, pd.DataFrame(stats_data), (offset_x, offset_y)


# 计算色区统计（支持选择使用原始坐标或移动后坐标）
def calculate_zone_statistics(df_dict, selected_bin_codes, selected_zones, color_zones=None,
                              move_center=False, offsets=(0, 0), use_original_coords=True):
    """计算每个色区的点数和占比，支持选择使用原始坐标或移动后坐标判断点是否在色区内"""
    if color_zones is None:
        color_zones = COLOR_ZONE_PRESETS["NCSP"]["zones"]  # 默认使用NCSP色区

    offset_x, offset_y = offsets
    stats = {}
    all_points_with_zones = []

    for file_name, df in df_dict.items():
        filtered_df = df[df['bin_code'].isin(selected_bin_codes)]
        if filtered_df.empty:
            continue

        total_points = len(filtered_df)
        file_stats = {
            'total_points': total_points,
            'zones': {}
        }

        # 为每个点添加所属色区信息
        zone_membership = []
        for _, row in filtered_df.iterrows():
            # 根据选择使用原始坐标或移动后的坐标判断点是否在色区内
            if use_original_coords:
                point = (row['ciex'], row['ciey'])  # 使用原始坐标
            else:
                point = (row['ciex'] + offset_x, row['ciey'] + offset_y)  # 使用移动后坐标

            in_zones = []
            for zone_name, zone_coords in color_zones.items():
                if zone_name in selected_zones and point_in_polygon(point, zone_coords):
                    in_zones.append(zone_name)
            zone_membership.append(", ".join(in_zones) if in_zones else "未命中")

        # 创建带色区信息的临时DataFrame
        temp_df = filtered_df.copy()
        temp_df['所属色区'] = zone_membership
        temp_df['数据类型'] = '原始数据' if use_original_coords else '移动后数据'
        temp_df['文件名'] = file_name
        all_points_with_zones.append(temp_df)

        # 统计每个色区的点数
        for zone_name in selected_zones:
            count = sum(1 for zones in zone_membership if zone_name in zones)
            percentage = (count / total_points) * 100 if total_points > 0 else 0
            file_stats['zones'][zone_name] = {
                'count': count,
                'percentage': percentage
            }

        # 统计未命中的点数
        count = sum(1 for zones in zone_membership if zones == "未命中")
        percentage = (count / total_points) * 100 if total_points > 0 else 0
        file_stats['zones']['未命中'] = {
            'count': count,
            'percentage': percentage
        }

        stats[file_name] = file_stats

    # 合并所有数据点
    if all_points_with_zones:
        combined_df = pd.concat(all_points_with_zones, ignore_index=True)
        return stats, combined_df
    else:
        return stats, pd.DataFrame()


# 计算产出分布统计
def calculate_production_statistics(df_dict, selected_bin_codes, color_zones, move_center=False,
                                    offsets=(0, 0), use_original_coords=True):
    """计算峰值波长、亮度、电压等参数的产出分布统计"""
    offset_x, offset_y = offsets
    all_data = []

    for file_name, df in df_dict.items():
        filtered_df = df[df['bin_code'].isin(selected_bin_codes)].copy()
        if filtered_df.empty:
            continue

        # 为每个点添加色区信息
        zone_membership = []
        for _, row in filtered_df.iterrows():
            if use_original_coords:
                point = (row['ciex'], row['ciey'])
            else:
                point = (row['ciex'] + offset_x, row['ciey'] + offset_y)

            in_zones = []
            for zone_name, zone_coords in color_zones.items():
                if point_in_polygon(point, zone_coords):
                    in_zones.append(zone_name)
            zone_membership.append(", ".join(in_zones) if in_zones else "未命中")

        filtered_df['所属色区'] = zone_membership
        filtered_df['文件名'] = file_name

        # 将参数值映射到对应的Bin区
        for param, config in PRODUCTION_BINS.items():
            bin_column = f"{param}_Bin"
            filtered_df[bin_column] = filtered_df[config['column']].apply(
                lambda x: value_to_bin(x, config['bins'])
            )

        all_data.append(filtered_df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


# 计算线性回归分析（基于移动后的坐标）
def calculate_linear_regression(df_dict, selected_bin_codes, move_center=False, offsets=(0, 0)):
    """计算CIE色坐标的线性回归分析，基于移动后的坐标"""
    offset_x, offset_y = offsets
    regression_results = {}

    for file_name, df in df_dict.items():
        filtered_df = df[df['bin_code'].isin(selected_bin_codes)]
        if len(filtered_df) < 2:  # 至少需要两个点进行线性回归
            continue

        # 如果启用了中心点移动，使用移动后的坐标计算
        if move_center:
            x = filtered_df['ciex'] + offset_x
            y = filtered_df['ciey'] + offset_y
        else:
            x = filtered_df['ciex']
            y = filtered_df['ciey']

        # 执行线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 计算残差平方和
        residuals = y - (slope * x + intercept)
        rss = np.sum(residuals ** 2)

        regression_results[file_name] = {
            '斜率': slope,
            '截距': intercept,
            'R² 值': r_value ** 2,
            'p 值': p_value,
            '标准误差': std_err,
            '残差平方和': rss
        }

    return regression_results


# 计算CIE色坐标差异
@st.cache_data(show_spinner=False)
def calculate_color_difference(ref_df, target_df, x_col='ciex', y_col='ciey', pos_cols=['PosX_Map', 'PosY_Map']):
    """计算目标数据与参考数据的色坐标差异"""
    # 合并参考数据和目标数据
    merged_df = pd.merge(
        ref_df[pos_cols + [x_col, y_col]].rename(columns={x_col: f'{x_col}_ref', y_col: f'{y_col}_ref'}),
        target_df[pos_cols + [x_col, y_col]].rename(columns={x_col: f'{x_col}_target', y_col: f'{y_col}_target'}),
        on=pos_cols,
        how='inner',
        suffixes=['_ref', '_target']
    )

    if merged_df.empty:
        return None

    # 计算欧几里得距离
    merged_df['color_distance'] = np.sqrt(
        (merged_df[f'{x_col}_target'] - merged_df[f'{x_col}_ref']) ** 2 +
        (merged_df[f'{y_col}_target'] - merged_df[f'{y_col}_ref']) ** 2
    )

    # 计算x和y的差异
    merged_df[f'{x_col}_diff'] = merged_df[f'{x_col}_target'] - merged_df[f'{x_col}_ref']
    merged_df[f'{y_col}_diff'] = merged_df[f'{y_col}_target'] - merged_df[f'{y_col}_ref']

    return merged_df


# 生成Mapping图（使用Plotly实现交互性）
@st.cache_data(show_spinner=False)
def generate_interactive_mapping_plot(df, value_col, title, fig_width=1000, fig_height=600,
                                      filter_outliers=False, ciex_range=None, ciey_range=None,
                                      special_markers=None, color_scale='viridis', product_type="NCSP",
                                      ncsp_region=None, cell_size=1.0,
                                      color_range=None, cluster_density=False, show_grid_subdivisions=True):
    """生成带交互功能的mapping图，包含优化的显示效果"""
    if df is None or df.empty:
        return None

    # 过滤异常点
    filtered_df = df.copy()
    if filter_outliers:
        if ciex_range:
            filtered_df = filtered_df[(filtered_df['ciex'] >= ciex_range[0]) &
                                      (filtered_df['ciex'] <= ciex_range[1])]
        if ciey_range:
            filtered_df = filtered_df[(filtered_df['ciey'] >= ciey_range[0]) &
                                      (filtered_df['ciey'] <= ciey_range[1])]

    if filtered_df.empty:
        return None

    # 确定使用的坐标列
    if product_type == "CSP":
        # 对CSP产品进行坐标归一化，使X和Y都从1开始
        min_x = filtered_df['pos_x'].min()
        filtered_df['pos_x_normalized'] = filtered_df['pos_x'] - min_x + 1

        min_y = filtered_df['pos_y'].min()
        filtered_df['pos_y_normalized'] = filtered_df['pos_y'] - min_y + 1

        x_col = 'pos_x_normalized'
        y_col = 'pos_y_normalized'
    else:  # NCSP
        x_col = 'PosX_Map'
        y_col = 'PosY_Map'

    # 针对NCSP产品进行区域筛选
    if product_type == "NCSP" and ncsp_region is not None:
        if ncsp_region == 1:
            filtered_df = filtered_df[filtered_df[x_col] <= 57]
        elif ncsp_region == 2:
            filtered_df = filtered_df[filtered_df[x_col] >= 78]

    # 计算数据密度用于动态调整点大小
    if cluster_density:
        # 使用KDE估算密度
        from scipy.stats import gaussian_kde
        xy = np.vstack([filtered_df[x_col], filtered_df[y_col]])
        z = gaussian_kde(xy)(xy)
        # 归一化密度值用于点大小调整
        z_scaled = (z - z.min()) / (z.max() - z.min())
        filtered_df['density'] = z_scaled
        size_base = 6 * cell_size
        size_column = size_base + (z_scaled * size_base * 2)  # 密度高的点稍大
    else:
        size_column = 8 * cell_size

    # 添加额外的悬停信息，根据产品类型显示不同内容
    hover_template = ""
    if product_type == "CSP":
        hover_template = (
            f"原始PosX: %{{customdata[0]}}<br>"
            f"原始PosY: %{{customdata[1]}}<br>"
            f"归一化PosX: %{{x:.1f}}<br>"
            f"归一化PosY: %{{y:.1f}}<br>"
            f"ciex: %{{customdata[2]:.4f}}<br>"
            f"ciey: %{{customdata[3]:.4f}}<br>"
            f"{value_col}: %{{marker.color:.4f}}<br>"
            f"bin_code: %{{customdata[4]}}"
        )
    else:
        hover_template = (
            f"PosX: %{{customdata[0]}}<br>"
            f"PosY: %{{customdata[1]}}<br>"
            f"ciex: %{{customdata[2]:.4f}}<br>"
            f"ciey: %{{customdata[3]:.4f}}<br>"
            f"{value_col}: %{{marker.color:.4f}}<br>"
            f"bin_code: %{{customdata[4]}}"
        )

    # 自动计算合适的颜色范围
    if color_range is None:
        # 使用95%置信区间避免极端值影响
        lower = filtered_df[value_col].quantile(0.025)
        upper = filtered_df[value_col].quantile(0.975)
        color_range = [lower, upper]

    # 创建基础散点图
    fig = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        color=value_col,
        color_continuous_scale=color_scale,
        range_color=color_range,
        title=title,
        labels={x_col: 'PosX' if product_type != "CSP" else '归一化PosX',
                y_col: 'PosY' if product_type != "CSP" else '归一化PosY',
                value_col: value_col},
        height=fig_height,
        width=fig_width,
        symbol_sequence=['square']
    )

    # 调整点大小和样式
    fig.update_traces(
        marker=dict(
            size=size_column,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)'),  # 更细的边框，半透明
            opacity=0.85
        ),
        customdata=list(zip(
            filtered_df.get('PosX_Map', filtered_df.get('pos_x')),
            filtered_df.get('PosY_Map', filtered_df.get('pos_y')),
            filtered_df['ciex'],
            filtered_df['ciey'],
            filtered_df['bin_code']
        )),
        hovertemplate=hover_template
    )

    # 添加网格细分
    if show_grid_subdivisions:
        x_min, x_max = filtered_df[x_col].min(), filtered_df[x_col].max()
        y_min, y_max = filtered_df[y_col].min(), filtered_df[y_col].max()

        # 添加次级网格线
        fig.update_layout(
            xaxis=dict(
                minor=dict(
                    tickvals=np.linspace(x_min, x_max, 21),  # 更多细分
                    gridcolor='rgba(200,200,200,0.2)',
                    gridwidth=1
                )
            ),
            yaxis=dict(
                minor=dict(
                    tickvals=np.linspace(y_min, y_max, 21),
                    gridcolor='rgba(200,200,200,0.2)',
                    gridwidth=1
                )
            )
        )

    # 应用特殊标记，优化显示效果
    if special_markers:
        for marker in special_markers:
            try:
                condition = marker['condition']
                color = marker['color']
                label = marker['label']

                # 筛选符合条件的数据
                special_df = filtered_df.query(condition)

                if not special_df.empty:
                    # 特殊标记悬停信息
                    special_hover = hover_template + f"<br><b>标记: {label}</b>"

                    # 添加特殊标记的散点
                    fig.add_trace(
                        go.Scatter(
                            x=special_df[x_col],
                            y=special_df[y_col],
                            mode='markers',
                            marker=dict(
                                size=12 * cell_size,
                                color=color,
                                symbol='diamond',
                                line=dict(width=2, color='black'),
                                opacity=0.9
                            ),
                            name=label,
                            customdata=list(zip(
                                special_df.get('PosX_Map', special_df.get('pos_x')),
                                special_df.get('PosY_Map', special_df.get('pos_y')),
                                special_df['ciex'],
                                special_df['ciey'],
                                special_df['bin_code']
                            )),
                            hovertemplate=special_hover,
                            # 确保特殊标记显示在最上层
                            layer='above'
                        )
                    )
            except Exception as e:
                st.warning(f"应用特殊标记时出错: {str(e)}")

    # 设置坐标轴范围和样式
    if product_type == "NCSP":
        if ncsp_region == 1:
            fig.update_xaxes(
                range=[0, 57],
                tickvals=[29],
                ticktext=["区域1 (1-57)"],
                showgrid=True,
                gridwidth=1.5,
                gridcolor='rgba(200,200,200,0.5)',
                minor=dict(showgrid=True)
            )
        elif ncsp_region == 2:
            fig.update_xaxes(
                range=[78, 134],
                tickvals=[106],
                ticktext=["区域2 (78-134)"],
                showgrid=True,
                gridwidth=1.5,
                gridcolor='rgba(200,200,200,0.5)',
                minor=dict(showgrid=True)
            )
        else:
            fig.update_xaxes(
                range=[0, 140],
                tickvals=[29, 106],
                ticktext=["区域1 (1-57)", "区域2 (78-134)"],
                showgrid=True,
                gridwidth=1.5,
                gridcolor='rgba(200,200,200,0.5)',
                minor=dict(showgrid=True)
            )

            # 添加区域分隔线
            fig.add_shape(
                type="line",
                x0=67.5, y0=0, x1=67.5, y1=55,
                line=dict(color="red", width=2, dash="dash"),
                name="区域分隔"
            )

            # 添加区域背景色（更浅的颜色）
            fig.add_shape(
                type="rect",
                x0=0, y0=0, x1=57, y1=55,
                line=dict(color="rgba(0,0,0,0)", width=0),
                fillcolor="rgba(0, 0, 255, 0.02)",
                layer="below"
            )
            fig.add_shape(
                type="rect",
                x0=78, y0=0, x1=134, y1=55,
                line=dict(color="rgba(0,0,0,0)", width=0),
                fillcolor="rgba(255, 0, 0, 0.02)",
                layer="below"
            )
    else:
        # CSP产品: 自适应调整X轴范围
        x_min, x_max = filtered_df[x_col].min(), filtered_df[x_col].max()
        x_margin = (x_max - x_min) * 0.05  # 添加5%的边距

        fig.update_xaxes(
            range=[x_min - x_margin, x_max + x_margin],
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.5)',
            minor=dict(showgrid=True),
            title='归一化PosX (从1开始)'
        )

        # 添加区域背景色
        fig.add_shape(
            type="rect",
            x0=x_min - x_margin, y0=filtered_df[y_col].min() - x_margin,
            x1=x_max + x_margin, y1=filtered_df[y_col].max() + x_margin,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor="rgba(0, 128, 0, 0.02)",
            layer="below"
        )

    # 设置Y轴范围，反转Y轴使数值小的在上方
    if product_type == "CSP":
        y_min, y_max = filtered_df[y_col].min(), filtered_df[y_col].max()
        y_margin = (y_max - y_min) * 0.05  # 添加5%的边距
        fig.update_yaxes(
            range=[y_max + y_margin, y_min - y_margin],  # 反转Y轴方向
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.5)',
            minor=dict(showgrid=True),
            title='归一化PosY (从1开始)'
        )
    else:
        # NCSP产品保持原有设置
        fig.update_yaxes(
            range=[55, 0],  # 反转Y轴方向
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.5)',
            minor=dict(showgrid=True)
        )

    # 优化交互性能和体验
    fig.update_layout(
        hovermode='closest',
        dragmode='zoom',
        selectdirection='h',
        margin=dict(l=40, r=40, t=60, b=40),  # 增加顶部边距放置标题
        transition_duration=300,  # 平滑过渡动画
        # # 新增：设置下载文件名（与标题一致，处理特殊字符）
        # toImageButtonOptions={
        #     'filename': title.replace(' ', '_').replace('.', '_').replace('/', '_'),  # 替换空格、点、斜杠等非法字符
        #     'format': 'png',  # 下载格式
        #     'height': fig_height,  # 高度与图表一致
        #     'width': fig_width,  # 宽度与图表一致
        #     'scale': 2  # 分辨率放大2倍，更清晰
        # },
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="SimHei",
            bordercolor="gray"  # 移除了不支持的borderwidth属性
        ),
        coloraxis_colorbar=dict(
            title=dict(text=value_col, font=dict(size=14)),
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside"
        ),
        # 添加图表说明
        annotations=[
            dict(
                text="提示: 框选区域可放大，双击可重置视图",
                x=0.5, y=1.02,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )

    # 添加颜色参考线（辅助判断数值）
    color_medians = np.percentile(filtered_df[value_col], [25, 50, 75])
    for median in color_medians:
        fig.add_hline(
            y=-1,  # 放置在图表外
            line_dash="dot",
            line_color="gray",
            annotation_text=f"{median:.4f}",
            annotation_position="right",
            annotation_font_color="gray"
        )

    return fig


# 更新图表的回调函数
def update_chart():
    st.session_state.chart_updated = True


# 目标中心点更新回调函数
def update_target_x():
    st.session_state.target_center_x = st.session_state.target_x_input


def update_target_y():
    st.session_state.target_center_y = st.session_state.target_y_input


# 添加回调函数来更新cell_size
def update_cell_size():
    st.session_state.cell_size = st.session_state.cell_size_slider


# 主函数
def main():
    st.title("CIE色点分析综合工具")

    # 初始化会话状态
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.dataframes = {}
        st.session_state.all_bin_codes = set()
        st.session_state.selected_bin_codes = []
        st.session_state.colors = {}
        st.session_state.chart_updated = False
        st.session_state.color_zone_preset = "NCSP"  # 默认选择NCSP
        st.session_state.csp_sub_type = "M"  # 默认选择一次模压
        st.session_state.selected_zones = list(COLOR_ZONE_PRESETS["NCSP"]["zones"].keys())
        st.session_state.move_center = False  # 中心点移动选项
        st.session_state.target_center_x = 0.2771  # 目标中心点X
        st.session_state.target_center_y = 0.26  # 目标中心点Y
        st.session_state.selected_total_zones = ['DK32', 'DK33', 'DK34', 'DK35', 'DL32', 'DL33', 'DL34', 'DL35']
        st.session_state.statistic_basis = "original"  # 统计依据：original-原始数据，moved-移动后数据
        st.session_state.production_data = None  # 存储产出分布数据，避免重复计算
        st.session_state.production_calculated = False  # 标记产出数据是否已计算
        st.session_state.special_markers = []  # 用于Mapping图特殊标记
        st.session_state.color_scale = 'viridis'  # 色阶方案
        st.session_state.product_type = "NCSP"  # 产品类型
        st.session_state.ncsp_region = None  # NCSP区域选择
        st.session_state.cell_size = 0.8  # 单元格大小
        st.session_state.show_slope_analysis = False  # 斜率分析开关
        st.session_state.slope_center = (0.2771, 0.26)  # 理想中心点默认值

    # 色区预设选择
    st.header("1. 色区预设与产品类型选择")
    col1, col2 = st.columns(2)
    with col1:
        preset_type = st.radio(
            "选择色区预设类型:",
            ["NCSP", "CSP"],
            index=0 if st.session_state.color_zone_preset == "NCSP" else 1,
            on_change=update_chart
        )
        st.session_state.color_zone_preset = preset_type

    with col2:
        product_type = st.selectbox(
            "产品类型",
            ["NCSP", "CSP"],
            index=0 if st.session_state.product_type == "NCSP" else 1,
            key="product_type_selector"
        )
        st.session_state.product_type = product_type

    # 如果选择了CSP，需要进一步选择子类型
    if preset_type == "CSP":
        csp_sub_type = st.radio(
            "选择CSP子类型:",
            ["M (一次模压)", "C (无水切割)"],
            index=0 if st.session_state.csp_sub_type == "M" else 1,
            on_change=update_chart
        )
        st.session_state.csp_sub_type = csp_sub_type.split()[0]  # 提取"M"或"C"

    # 根据选择的预设获取色区数据
    if preset_type == "NCSP":
        color_zones = COLOR_ZONE_PRESETS["NCSP"]["zones"]
        preset_name = COLOR_ZONE_PRESETS["NCSP"]["name"]
    else:  # CSP
        sub_type = st.session_state.csp_sub_type
        color_zones = COLOR_ZONE_PRESETS["CSP"]["sub_types"][sub_type]["zones"]
        preset_name = f"CSP色区 ({COLOR_ZONE_PRESETS['CSP']['sub_types'][sub_type]['name']})"

    st.info(f"当前使用的色区预设: {preset_name}")

    # 文件上传部分
    st.header("2. 文件上传")

    uploaded_files = st.file_uploader(
        "上传分光数据文件 (支持XLSX, CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 添加编码选择
        encoding = st.selectbox(
            "选择 CSV 文件编码",
            ['gbk', 'utf-8', 'gb2312', 'iso-8859-1'],
            index=0
        )

        # 加载数据
        with st.spinner("正在加载数据..."):
            start_time = time.time()
            st.session_state.uploaded_files = uploaded_files
            st.session_state.dataframes = {}
            for file in uploaded_files:
                # 调用更新后的load_data函数，传入product_type参数
                df = load_data(file, st.session_state.product_type, encoding)
                if df is not None:
                    st.session_state.dataframes[file.name] = df
            load_time = time.time() - start_time
            st.success(f"成功加载 {len(st.session_state.dataframes)} 个文件，耗时 {load_time:.2f} 秒")

        if st.session_state.dataframes:
            # 获取所有bin_code
            all_bin_codes = set()
            for df in st.session_state.dataframes.values():
                all_bin_codes.update(df['bin_code'].unique())
            st.session_state.all_bin_codes = sorted(all_bin_codes)

            # 筛选条件设置
            st.header("3. 筛选条件设置")

            # 确保选中的bin_code存在于当前选项中
            valid_selected_bin_codes = []
            if st.session_state.selected_bin_codes:
                for code in st.session_state.selected_bin_codes:
                    if code in st.session_state.all_bin_codes:
                        valid_selected_bin_codes.append(code)

            # 如果没有有效选项，则默认选择所有bin_code
            if not valid_selected_bin_codes and st.session_state.all_bin_codes:
                valid_selected_bin_codes = st.session_state.all_bin_codes

            selected_bin_codes = st.multiselect(
                "选择要包含的bin_code:",
                st.session_state.all_bin_codes,
                default=valid_selected_bin_codes,
                on_change=update_chart
            )
            st.session_state.selected_bin_codes = selected_bin_codes

            # 颜色选择
            st.subheader("数据颜色设置")
            for i, file_name in enumerate(st.session_state.dataframes.keys()):
                default_color = color_list[i % len(color_list)]
                color = st.color_picker(
                    f"选择 {file_name} 的颜色:",
                    default_color,
                    key=f"color_{file_name}",
                    on_change=update_chart
                )
                st.session_state.colors[file_name] = color

            # 使用选项卡组织不同的分析功能
            tab1, tab2, tab3, tab4 = st.tabs(["CIE色区统计", "色区详细统计", "产出分布统计", "Mapping图分析"])

            # 1. CIE色区统计选项卡
            with tab1:
                st.header("CIE色区分布图")

                # 图表设置
                st.subheader("图表设置")
                col1, col2 = st.columns(2)
                with col1:
                    fig_width = st.slider("图表宽度 (像素)", 800, 2000, 1200, on_change=update_chart,
                                          key="scatter_width")
                    fig_height = st.slider("图表高度 (像素)", 600, 1500, 800, on_change=update_chart,
                                           key="scatter_height")
                    point_size = st.slider("点大小", 1, 50, 6, on_change=update_chart, key="scatter_point_size")
                    alpha = st.slider("点透明度", 0.1, 1.0, 0.8, on_change=update_chart, key="scatter_alpha")

                with col2:
                    title = st.text_input("图表标题", f"{preset_name} CIE色区分布图", on_change=update_chart,
                                          key="scatter_title")
                    x_label = st.text_input("X轴标签", "ciex", on_change=update_chart, key="scatter_x_label")
                    y_label = st.text_input("Y轴标签", "ciey", on_change=update_chart, key="scatter_y_label")
                    show_grid = st.checkbox("显示网格", True, on_change=update_chart, key="scatter_grid")

                # 中心点移动设置
                st.subheader("中心点移动设置")
                move_center = st.checkbox(
                    "启用中心点移动",
                    value=st.session_state.move_center,
                    on_change=update_chart,
                    key="move_center_checkbox"
                )
                st.session_state.move_center = move_center

                target_center = (st.session_state.target_center_x, st.session_state.target_center_y)
                if move_center:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.number_input(
                            "目标中心点 X",
                            value=st.session_state.target_center_x,
                            step=0.0001,
                            format="%.4f",
                            on_change=update_target_x,
                            key="target_x_input"
                        )
                    with col2:
                        st.number_input(
                            "目标中心点 Y",
                            value=st.session_state.target_center_y,
                            step=0.0001,
                            format="%.4f",
                            on_change=update_target_y,
                            key="target_y_input"
                        )
                    target_center = (st.session_state.target_center_x, st.session_state.target_center_y)
                    st.info(
                        f"将根据目标中心点 ({target_center[0]:.4f}, {target_center[1]:.4f}) 计算补偿系数并移动数据点")
                    st.info("注意：启用中心点移动后，仅数据点会移动，色区位置保持固定不变")
                # 1. 坐标轴范围设置（修改：新增固定比例选项）
                st.subheader("坐标轴范围设置")
                # 新增：比例选择器
                axis_scale_option = st.radio(
                    "选择坐标轴比例",
                    ["自动范围", "固定比例（x:0.0055/格, y:0.01/格）"],
                    key="axis_scale_option",
                    on_change=update_chart
                )

                # 计算所有数据的范围（原有逻辑保留，新增比例判断）
                all_ciex = []
                all_ciey = []
                for df in st.session_state.dataframes.values():
                    filtered_df = df[df['bin_code'].isin(selected_bin_codes)]
                    if not filtered_df.empty:
                        if move_center:
                            all_ciex.extend(
                                (filtered_df['ciex'] + (target_center[0] - filtered_df['ciex'].mean())).tolist())
                            all_ciey.extend(
                                (filtered_df['ciey'] + (target_center[1] - filtered_df['ciey'].mean())).tolist())
                        else:
                            all_ciex.extend(filtered_df['ciex'].tolist())
                            all_ciey.extend(filtered_df['ciey'].tolist())
                for zone_coords in color_zones.values():
                    for x, y in zone_coords:
                        all_ciex.append(x)
                        all_ciey.append(y)

                if all_ciex and all_ciey:
                    # 新增：根据比例选项计算范围
                    if axis_scale_option == "固定比例（x:0.0055/格, y:0.01/格）":
                        x_range, y_range = calculate_fixed_ratio_range(all_ciex, all_ciey)
                    else:  # 自动范围（原有逻辑）
                        x_min, x_max = min(all_ciex), max(all_ciex)
                        y_min, y_max = min(all_ciey), max(all_ciey)
                        x_margin = (x_max - x_min) * 0.1
                        y_margin = (y_max - y_min) * 0.1
                        x_range = [x_min - x_margin, x_max + x_margin]
                        y_range = [y_min - y_margin, y_max + y_margin]

                    # 手动调整（原有逻辑保留）
                    if st.checkbox("手动调整范围", value=False, key="manual_range_checkbox", on_change=update_chart):
                        col1, col2 = st.columns(2)
                        with col1:
                            x_min_custom = st.number_input("X轴最小值", value=round(x_range[0], 4), step=0.0001,
                                                           format="%.4f", key="scatter_x_min")
                            x_max_custom = st.number_input("X轴最大值", value=round(x_range[1], 4), step=0.0001,
                                                           format="%.4f", key="scatter_x_max")
                        with col2:
                            y_min_custom = st.number_input("Y轴最小值", value=round(y_range[0], 4), step=0.0001,
                                                           format="%.4f", key="scatter_y_min")
                            y_max_custom = st.number_input("Y轴最大值", value=round(y_range[1], 4), step=0.0001,
                                                           format="%.4f", key="scatter_y_max")
                        x_range = [x_min_custom, x_max_custom]
                        y_range = [y_min_custom, y_max_custom]
                else:
                    st.warning("没有足够的数据来计算坐标轴范围")
                    x_range = [0, 0.044]  # 固定比例默认范围
                    y_range = [0, 0.05]

                # 2. 新增：斜率分析设置（在“色区选择”前插入）
                st.subheader("斜率分析设置")
                show_slope_analysis = st.checkbox("显示理想斜率线", value=False, key="show_slope_analysis",
                                                  on_change=update_chart)
                slope_line_info = None  # 存储斜率线信息（方程+坐标）

                if show_slope_analysis:
                    # 理想中心点设置（默认0.2771, 0.26）
                    st.markdown("#### 理想中心点")
                    col1, col2 = st.columns(2)
                    with col1:
                        slope_center_x = st.number_input(
                            "X坐标", value=0.2771, step=0.0001, format="%.4f", key="slope_center_x"
                        )
                    with col2:
                        slope_center_y = st.number_input(
                            "Y坐标", value=0.26, step=0.0001, format="%.4f", key="slope_center_y"
                        )
                    slope_center = (slope_center_x, slope_center_y)

                    # 找到包含中心点的色区
                    zone_name, zone_coords = find_zone_containing_point(slope_center, color_zones)
                    if zone_name and zone_coords:
                        st.success(f"找到包含中心点的色区：{zone_name}")
                        # 计算色区的正斜率
                        positive_slope = calculate_parallelogram_positive_slopes(zone_coords)
                        if positive_slope:
                            st.info(f"色区正斜率：{positive_slope:.4f}")
                            # 计算直线方程和坐标
                            slope_line_info = get_slope_line_params(slope_center, positive_slope)
                        else:
                            st.warning("该色区无正斜率边，请选择其他包含中心点的色区")
                    else:
                        st.warning("理想中心点不在任何色区内，请调整中心点坐标")


                # 色区选择
                st.subheader("色区选择")
                # 确保选中的色区存在于当前预设中
                valid_selected_zones = []
                if st.session_state.selected_zones:
                    for zone in st.session_state.selected_zones:
                        if zone in color_zones:
                            valid_selected_zones.append(zone)

                # 如果没有有效选项，则默认选择所有色区
                if not valid_selected_zones:
                    valid_selected_zones = list(color_zones.keys())

                selected_zones = st.multiselect(
                    "选择要显示的色区:",
                    list(color_zones.keys()),
                    default=valid_selected_zones,
                    on_change=update_chart
                )
                st.session_state.selected_zones = selected_zones

                # 绘制CIE散点图
                st.subheader("CIE色区分布图")
                with st.spinner("正在生成图表..."):
                    start_time = time.time()
                    fig, stats_df, offsets = generate_interactive_cie_plot_with_zones(
                        st.session_state.dataframes,
                        selected_bin_codes,
                        st.session_state.colors,
                        title,
                        fig_width,
                        fig_height,
                        point_size,
                        alpha,
                        x_label,
                        y_label,
                        show_grid,
                        x_range,
                        y_range,
                        selected_zones,
                        color_zones,
                        move_center,
                        target_center,
                        show_slope_line = show_slope_analysis,  # 新增
                        slope_line_info = slope_line_info  # 新增
                    )
                    plot_time = time.time() - start_time

                    # 显示图表和统计数据
                    # 修改后
                    # 获取所有上传的文件名（去除扩展名）
                    file_names = [os.path.splitext(name)[0] for name in st.session_state.dataframes.keys()]
                    # 拼接文件名（最多显示3个，避免过长）
                    combined_name = "_".join(file_names[:3]) + ("..." if len(file_names) > 3 else "")
                    # 配置下载文件名
                    config = {
                        'toImageButtonOptions': {
                            'filename': f"CIE色点分布图_{combined_name}",
                            'format': 'png',  # 可选项：'png'、'svg'、'jpeg'、'webp'
                            'height': fig_height,
                            'width': fig_width,
                            'scale': 2  # 分辨率缩放
                        }
                    }
                    st.plotly_chart(fig, config=config, use_container_width=True)


                    # 显示统计数据
                    st.subheader("基本统计数据")
                    if not stats_df.empty:
                        # 显示补偿系数（如果启用了中心点移动）
                        if move_center and offsets != (0, 0):
                            st.info(f"应用的补偿系数: X偏移 = {offsets[0]:.6f}, Y偏移 = {offsets[1]:.6f}")
                        st.dataframe(stats_df.round(4))
                    else:
                        st.info("没有足够的数据生成统计信息")

                    st.markdown("### 使用说明")
                    st.markdown("- 鼠标悬停：查看点的详细数值")
                    st.markdown("- 滚轮：缩放图表")
                    st.markdown("- 拖动：平移图表")
                    st.markdown("- 框选：局部放大")
                    st.markdown("- 双击：重置视图")
                    st.markdown("- 点击图例：显示/隐藏特定材料或色区")
                    st.markdown("- 右键点击：下载图表为PNG/SVG")

            # 2. 色区详细统计选项卡
            with tab2:
                st.header("色区详细统计")

                # 添加统计依据选择
                st.subheader("统计设置")
                statistic_basis = st.radio(
                    "选择统计依据:",
                    ["原始数据", "移动后数据"],
                    index=0 if st.session_state.statistic_basis == "original" else 1,
                    key="statistic_basis_radio"
                )
                st.session_state.statistic_basis = "original" if statistic_basis == "原始数据" else "moved"
                use_original_coords = (st.session_state.statistic_basis == "original")

                # 生成色区统计
                if st.button("生成色区详细统计", key="generate_zone_stats"):
                    with st.spinner(f"正在计算{'原始' if use_original_coords else '移动后'}数据的色区统计..."):
                        # 重新计算图表以获取最新的偏移值
                        _, _, offsets = generate_interactive_cie_plot_with_zones(
                            st.session_state.dataframes,
                            selected_bin_codes,
                            st.session_state.colors,
                            "",  # 标题不重要
                            800, 400,  # 尺寸不重要
                            6, 0.8,  # 点大小和透明度不重要
                            "ciex", "ciey", True,  # 轴标签和网格不重要
                            [0, 1], [0, 1],  # 范围不重要
                            selected_zones,
                            color_zones,
                            st.session_state.move_center,
                            (st.session_state.target_center_x, st.session_state.target_center_y)
                        )

                        zone_stats, points_with_zones = calculate_zone_statistics(
                            st.session_state.dataframes,
                            selected_bin_codes,
                            selected_zones,
                            color_zones,
                            st.session_state.move_center,
                            offsets,
                            use_original_coords  # 传递选择的统计依据
                        )

                        # 显示统计结果
                        st.subheader(f"色区落Bin率统计（基于{statistic_basis}）")

                        # 总占比色区选择
                        st.subheader("总占比统计设置")
                        # 过滤出当前色区预设中存在的默认总占比色区
                        # 修改后
                        if st.session_state.color_zone_preset == "CSP":
                            # CSP类型：根据子类型自动选择默认色区
                            sub_type = st.session_state.csp_sub_type
                            if sub_type == "M":
                                # 一次模压：默认所有带_M后缀的色区
                                valid_default_zones = [zone for zone in color_zones.keys() if zone.endswith("_M")]
                            else:  # C
                                # 无水切割：默认所有带_C后缀的色区
                                valid_default_zones = [zone for zone in color_zones.keys() if zone.endswith("_C")]
                        else:
                            # NCSP类型：保持原有逻辑
                            valid_default_zones = [zone for zone in st.session_state.selected_total_zones if
                                                   zone in color_zones]
                        selected_total_zones = st.multiselect(
                            "选择要计算总占比的色区组合:",
                            list(color_zones.keys()),
                            default=valid_default_zones,
                            key="total_zones_selector"
                        )
                        st.session_state.selected_total_zones = selected_total_zones

                        # 创建统计表格
                        for file_name, stats in zone_stats.items():
                            st.subheader(f"文件: {file_name}")
                            st.text(f"总点数: {stats['total_points']}")

                            # 创建数据框显示统计结果
                            stats_data = []
                            for zone_name, zone_stats in stats['zones'].items():
                                stats_data.append({
                                    '色区': zone_name,
                                    '点数': zone_stats['count'],
                                    '占比(%)': round(zone_stats['percentage'], 2)
                                })

                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df)

                            # 原“生成色区详细统计”按钮及相关逻辑删除，替换为以下代码：
                            st.subheader(f"文件: {file_name}")
                            st.text(f"总点数: {stats['total_points']}")

                            # 生成对应图表
                            stats_data = []
                            for zone_name, zone_stats in stats['zones'].items():
                                stats_data.append({
                                    '色区': zone_name,
                                    '点数': zone_stats['count'],
                                    '占比(%)': round(zone_stats['percentage'], 2)
                                })
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df)

                            # 生成柱状图（默认）
                            stats_data = []
                            for zone_name, zone_stats in stats['zones'].items():
                                stats_data.append({
                                    '色区': zone_name,
                                    '点数': zone_stats['count'],
                                    '占比(%)': round(zone_stats['percentage'], 2)
                                })
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df)

                            # 直接生成柱状图
                            fig = px.bar(
                                stats_df,
                                x='色区',
                                y='占比(%)',
                                title=f'{file_name} 各色区占比（基于{statistic_basis}）',
                                text='占比(%)',
                                color='色区',
                                color_discrete_sequence=color_list,
                                hover_data=['点数', '占比(%)']
                            )
                            fig.update_traces(
                                hovertemplate='色区: %{x}<br>数量: %{customdata[0]}<br>占比: %{customdata[1]:.2f}%<extra></extra>'
                            )
                            fig.update_layout(
                                yaxis=dict(title='占比(%)', range=[0, 100]),
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)


                            # 计算并显示总占比
                            if selected_total_zones and stats['total_points'] > 0:
                                total_count = 0
                                for zone in selected_total_zones:
                                    if zone in stats['zones']:
                                        total_count += stats['zones'][zone]['count']

                                total_percentage = round((total_count / stats['total_points']) * 100, 2)

                                st.subheader(f"所选色区总占比统计（基于{statistic_basis}）")
                                total_stats_df = pd.DataFrame([{
                                    '所选色区': ', '.join(selected_total_zones),
                                    '总点数': total_count,
                                    '总占比(%)': total_percentage
                                }])
                                st.dataframe(total_stats_df)

                                # 创建总占比可视化
                                fig_total = px.pie(
                                    names=['所选色区总和', '其他'],
                                    values=[total_percentage, 100 - total_percentage],
                                    title=f'{file_name} 所选色区总占比（基于{statistic_basis}）',
                                    color_discrete_sequence=['#2ca02c', '#d62728']
                                )
                                fig_total.update_traces(textinfo='label+percent')
                                st.plotly_chart(fig_total, use_container_width=True)

                        # 显示带有色区信息的数据样本
                        st.subheader(f"色区数据样本（基于{statistic_basis}）")
                        if not points_with_zones.empty:
                            st.dataframe(
                                points_with_zones[
                                    ['文件名', 'PosX_Map' if 'PosX_Map' in points_with_zones.columns else 'pos_x',
                                     'PosY_Map' if 'PosY_Map' in points_with_zones.columns else 'pos_y',
                                     'ciex', 'ciey', 'bin_code', '所属色区']].head(
                                    100))
                        else:
                            st.info("没有数据可显示")

                        # 计算并显示线性回归分析
                        st.subheader(f"CIE色坐标线性回归分析（基于{statistic_basis}）")
                        regression_results = calculate_linear_regression(
                            st.session_state.dataframes,
                            selected_bin_codes,
                            st.session_state.move_center,
                            offsets
                        )

                        if regression_results:
                            for file_name, results in regression_results.items():
                                st.markdown(f"#### {file_name} 的线性回归结果")
                                results_df = pd.DataFrame([results])
                                st.dataframe(results_df.round(6))

                                # 可视化线性回归结果
                                filtered_df = st.session_state.dataframes[file_name]
                                filtered_df = filtered_df[filtered_df['bin_code'].isin(selected_bin_codes)]

                                if not filtered_df.empty:
                                    # 应用中心点移动（如果启用）
                                    if st.session_state.move_center and not use_original_coords:
                                        x = filtered_df['ciex'] + offsets[0]
                                        y = filtered_df['ciey'] + offsets[1]
                                    else:
                                        x = filtered_df['ciex']
                                        y = filtered_df['ciey']

                                    # 创建散点图和回归线
                                    fig = px.scatter(
                                        x=x,
                                        y=y,
                                        title=f'{file_name} 的CIE色坐标线性回归（基于{statistic_basis}）',
                                        labels={'x': 'ciex', 'y': 'ciey'},
                                        color_discrete_sequence=[st.session_state.colors[file_name]]
                                    )

                                    # 添加回归线
                                    slope = results['斜率']
                                    intercept = results['截距']
                                    x_range_reg = [x.min(), x.max()]
                                    y_range_reg = [slope * x_range_reg[0] + intercept,
                                                   slope * x_range_reg[1] + intercept]

                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_range_reg,
                                            y=y_range_reg,
                                            mode='lines',
                                            line=dict(color='red', width=2),
                                            name=f'回归线 (y = {slope:.6f}x + {intercept:.6f})'
                                        )
                                    )

                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("数据点不足，无法进行线性回归分析")

            # 3. 产出分布统计选项卡
            with tab3:
                st.header("产出分布统计")
                st.info(f"该统计基于{statistic_basis}的色区判断结果")

                # 重置计算状态的按钮
                if st.button("重置产出分布统计", key="reset_production_stats"):
                    st.session_state.production_data = None
                    st.session_state.production_calculated = False
                    st.success("已重置产出分布统计数据")

                # 生成产出分布统计
                if st.button("生成产出分布统计",
                             key="generate_production_stats") or st.session_state.production_calculated:
                    # 如果数据已计算且不是首次点击，则直接使用缓存数据
                    if not st.session_state.production_calculated or st.session_state.production_data is None:
                        with st.spinner(f"正在计算产出分布统计..."):
                            # 重新计算图表以获取最新的偏移值
                            _, _, offsets = generate_interactive_cie_plot_with_zones(
                                st.session_state.dataframes,
                                selected_bin_codes,
                                st.session_state.colors,
                                "",  # 标题不重要
                                800, 400,  # 尺寸不重要
                                6, 0.8,  # 点大小和透明度不重要
                                "ciex", "ciey", True,  # 轴标签和网格不重要
                                [0, 1], [0, 1],  # 范围不重要
                                selected_zones,
                                color_zones,
                                st.session_state.move_center,
                                (st.session_state.target_center_x, st.session_state.target_center_y)
                            )

                            # 获取色区统计依据
                            use_original_coords = (st.session_state.statistic_basis == "original")

                            # 计算产出分布统计
                            production_data = calculate_production_statistics(
                                st.session_state.dataframes,
                                selected_bin_codes,
                                color_zones,
                                st.session_state.move_center,
                                offsets,
                                use_original_coords
                            )

                            # 存储计算结果到会话状态
                            st.session_state.production_data = production_data
                            st.session_state.production_calculated = True
                    else:
                        production_data = st.session_state.production_data

                    if production_data.empty:
                        st.warning("没有足够的数据生成产出分布统计")
                        return

                    # 按文件分别统计
                    for file_idx, file_name in enumerate(st.session_state.dataframes.keys()):
                        st.subheader(f"文件: {file_name}")
                        file_data = production_data[production_data['文件名'] == file_name]
                        total_points = len(file_data)
                        st.text(f"总样本数: {total_points}")

                        if total_points == 0:
                            st.info("该文件没有符合筛选条件的数据")
                            continue

                        # 1. 峰值波长分布
                        st.subheader("1. 峰值波长分布")
                        wavelength_config = PRODUCTION_BINS["Wavelength"]

                        # 统计数据
                        wavelength_counts = file_data['Wavelength_Bin'].value_counts().reindex(
                            wavelength_config['order'], fill_value=0
                        )
                        wavelength_percent = [round((count / total_points * 100), 2) for count in
                                              wavelength_counts.values]

                        # 创建数据框
                        wavelength_stats = pd.DataFrame({
                            'Bin区': wavelength_counts.index,
                            f'范围({wavelength_config["units"]})': [
                                f"{wavelength_config['bins'][bin][0]}-{wavelength_config['bins'][bin][1]}"
                                for bin in wavelength_counts.index
                            ],
                            '数量': wavelength_counts.values,
                            '占比(%)': wavelength_percent
                        })
                        st.dataframe(wavelength_stats)

                        # 创建可视化图表
                        fig = px.bar(
                            wavelength_stats,
                            x='Bin区',
                            y='占比(%)',
                            title=f'{file_name} 峰值波长分布',
                            color='Bin区',
                            color_discrete_sequence=color_list[:len(wavelength_config['order'])],
                            hover_data=['数量', '占比(%)', f'范围({wavelength_config["units"]})']
                        )
                        fig.update_traces(
                            hovertemplate='Bin区: %{x}<br>范围: %{customdata[2]}<br>数量: %{customdata[0]}<br>占比: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        fig.update_layout(
                            yaxis=dict(title='占比(%)', range=[0, 100]),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 2. 亮度分布
                        st.subheader("2. 亮度分布")
                        brightness_config = PRODUCTION_BINS["Brightness"]

                        # 统计数据
                        brightness_counts = file_data['Brightness_Bin'].value_counts().reindex(
                            brightness_config['order'], fill_value=0
                        )
                        brightness_percent = [round((count / total_points * 100), 2) for count in
                                              brightness_counts.values]

                        # 创建数据框
                        brightness_stats = pd.DataFrame({
                            'Bin区': brightness_counts.index,
                            f'范围({brightness_config["units"]})': [
                                f"{brightness_config['bins'][bin][0]}-{brightness_config['bins'][bin][1]}"
                                for bin in brightness_counts.index
                            ],
                            '数量': brightness_counts.values,
                            '占比(%)': brightness_percent
                        })
                        st.dataframe(brightness_stats)

                        # 创建可视化图表
                        fig = px.bar(
                            brightness_stats,
                            x='Bin区',
                            y='占比(%)',
                            title=f'{file_name} 亮度分布',
                            color='Bin区',
                            color_discrete_sequence=color_list[:len(brightness_config['order'])],
                            hover_data=['数量', '占比(%)', f'范围({brightness_config["units"]})']
                        )
                        fig.update_traces(
                            hovertemplate='Bin区: %{x}<br>范围: %{customdata[2]}<br>数量: %{customdata[0]}<br>占比: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        fig.update_layout(
                            yaxis=dict(title='占比(%)', range=[0, 100]),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 3. 电压分布
                        st.subheader("3. 电压分布")
                        voltage_config = PRODUCTION_BINS["Voltage"]

                        # 统计数据
                        voltage_counts = file_data['Voltage_Bin'].value_counts().reindex(
                            voltage_config['order'], fill_value=0
                        )
                        voltage_percent = [round((count / total_points * 100), 2) for count in voltage_counts.values]

                        # 创建数据框
                        voltage_stats = pd.DataFrame({
                            'Bin区': voltage_counts.index,
                            f'范围({voltage_config["units"]})': [
                                f"{voltage_config['bins'][bin][0]}-{voltage_config['bins'][bin][1]}"
                                for bin in voltage_counts.index
                            ],
                            '数量': voltage_counts.values,
                            '占比(%)': voltage_percent
                        })
                        st.dataframe(voltage_stats)

                        # 创建可视化图表
                        fig = px.bar(
                            voltage_stats,
                            x='Bin区',
                            y='占比(%)',
                            title=f'{file_name} 电压分布',
                            color='Bin区',
                            color_discrete_sequence=color_list[:len(voltage_config['order'])],
                            hover_data=['数量', '占比(%)', f'范围({voltage_config["units"]})']
                        )
                        fig.update_traces(
                            hovertemplate='Bin区: %{x}<br>范围: %{customdata[2]}<br>数量: %{customdata[0]}<br>占比: %{customdata[1]:.2f}%<extra></extra>'
                        )
                        fig.update_layout(
                            yaxis=dict(title='占比(%)', range=[0, 100]),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 4. 色区与参数的交叉分析
                        st.subheader("4. 色区与参数的交叉分析")

                        # 使用文件索引确保key的唯一性
                        param_to_analyze = st.selectbox(
                            "选择要与色区进行交叉分析的参数:",
                            ["峰值波长", "亮度", "电压"],
                            key=f"param_selector_{file_idx}_{file_name}"
                        )

                        # 映射参数到对应的列名
                        param_mapping = {
                            "峰值波长": "Wavelength_Bin",
                            "亮度": "Brightness_Bin",
                            "电压": "Voltage_Bin"
                        }
                        param_bin_column = param_mapping[param_to_analyze]

                        # 获取参数配置
                        if param_to_analyze == "峰值波长":
                            param_config = PRODUCTION_BINS["Wavelength"]
                        elif param_to_analyze == "亮度":
                            param_config = PRODUCTION_BINS["Brightness"]
                        else:  # 电压
                            param_config = PRODUCTION_BINS["Voltage"]

                        try:
                            # 创建交叉表
                            cross_df = pd.crosstab(
                                file_data['所属色区'],
                                file_data[param_bin_column],
                                margins=True,
                                margins_name='总计'
                            )

                            # 按顺序重新排列列
                            ordered_columns = param_config['order'] + ['总计']
                            cross_df = cross_df.reindex(columns=ordered_columns, fill_value=0)

                            # 显示交叉表
                            st.dataframe(cross_df)

                            # 创建交叉分析热图（排除总计行和列）
                            if len(cross_df) > 1 and len(cross_df.columns) > 1:
                                heatmap_data = cross_df.drop('总计').drop(columns='总计')

                                fig = px.imshow(
                                    heatmap_data,
                                    text_auto=True,
                                    title=f'{file_name} 色区与{param_to_analyze}的交叉分布',
                                    color_continuous_scale='YlOrRd'
                                )
                                fig.update_layout(
                                    xaxis_title=param_to_analyze,
                                    yaxis_title='色区'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("数据量不足，无法生成交叉分析热图")
                        except Exception as e:
                            st.error(f"生成交叉分析时出错: {str(e)}")

                        # 5. bin产出分布统计（新增）
                        st.subheader("5. bin产出分布统计 (1-80排序)")

                        # 定义1到80的bin号
                        bins_1_to_80 = list(range(1, 81))  # 生成1-80的列表

                        # 1. 定义需要统计的bin号范围（1-80）
                        bins_1_to_80 = list(range(1, 81))  # 生成[1,2,...,80]的列表

                        # 2. 统计bin列，并按1-80的范围对齐，缺失的bin号计数填充为0
                        bin_counts = df['bin'].value_counts().reindex(bins_1_to_80, fill_value=0)
                        bin_percent = [round((count / total_points * 100), 2) for count in bin_counts.values]

                        # 创建统计表格（只包含bin号、计数、占比）
                        bin_stats = pd.DataFrame({
                            'bin号': bins_1_to_80,
                            '计数': bin_counts.values,
                            '占比(%)': bin_percent
                        })
                        # 按bin号升序排列（确保1-80顺序）
                        bin_stats = bin_stats.sort_values('bin号').reset_index(drop=True)
                        st.dataframe(bin_stats)  # 显示表格

                        # 生成柱状图展示bin分布
                        fig = px.bar(
                            bin_stats,
                            x='bin号',
                            y='计数',
                            title=f'{file_name} bin产出分布 (1-80)',
                            color='计数',
                            color_continuous_scale='Viridis',
                            hover_data=['bin号', '计数', '占比(%)']
                        )
                        fig.update_layout(
                            xaxis=dict(tickmode='linear', dtick=5),  # 每5个bin号显示一个刻度
                            xaxis_title='bin号',
                            yaxis_title='计数'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 生成饼图（仅显示有数据的bin，避免80个类别拥挤）
                        filtered_bin_stats = bin_stats[bin_stats['计数'] > 0]
                        if not filtered_bin_stats.empty:
                            fig = px.pie(
                                filtered_bin_stats,
                                values='计数',
                                names='bin号',
                                title=f'{file_name} bin产出分布（仅显示有数据的bin）',
                                color='bin号',
                                color_discrete_sequence=color_list  # 使用预设颜色列表
                            )
                            # 饼图悬停信息：包含计数、占比
                            fig.update_traces(
                                hovertemplate='bin号: %{label}<br>计数: %{value}<br>占比: %{percent}<extra></extra>',
                                customdata=list(zip(filtered_bin_stats['占比(%)']))
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("当前筛选条件下没有bin数据")


                        # 6. 产出分布综合分析
                        st.subheader("6. 产出分布综合分析")

                        # 计算各参数的主要分布区域
                        try:
                            # 峰值波长主要分布区
                            wavelength_main_bin = wavelength_counts.idxmax()
                            wavelength_main_percent = wavelength_percent[
                                wavelength_counts.index.get_loc(wavelength_main_bin)]

                            # 亮度主要分布区
                            brightness_main_bin = brightness_counts.idxmax()
                            brightness_main_percent = brightness_percent[
                                brightness_counts.index.get_loc(brightness_main_bin)]

                            # 电压主要分布区
                            voltage_main_bin = voltage_counts.idxmax()
                            voltage_main_percent = voltage_percent[voltage_counts.index.get_loc(voltage_main_bin)]

                            # 主要色区
                            color_zone_counts = file_data['所属色区'].value_counts()
                            main_color_zone = color_zone_counts.index[0]
                            main_color_percent = round((color_zone_counts[0] / total_points * 100), 2)

                            # 显示综合分析结果
                            st.markdown(f"**主要分布区域分析**")
                            st.markdown(f"- 主要峰值波长Bin区: {wavelength_main_bin} "
                                        f"({wavelength_config['bins'][wavelength_main_bin][0]}-"
                                        f"{wavelength_config['bins'][wavelength_main_bin][1]} {wavelength_config['units']})，"
                                        f"占比 {wavelength_main_percent}%")
                            st.markdown(f"- 主要亮度Bin区: {brightness_main_bin} "
                                        f"({brightness_config['bins'][brightness_main_bin][0]}-"
                                        f"{brightness_config['bins'][brightness_main_bin][1]} {brightness_config['units']})，"
                                        f"占比 {brightness_main_percent}%")
                            st.markdown(f"- 主要电压Bin区: {voltage_main_bin} "
                                        f"({voltage_config['bins'][voltage_main_bin][0]}-"
                                        f"{voltage_config['bins'][voltage_main_bin][1]} {voltage_config['units']})，"
                                        f"占比 {voltage_main_percent}%")
                            st.markdown(f"- 主要色区: {main_color_zone}，占比 {main_color_percent}%")

                            # 生成综合分布雷达图
                            radar_data = []
                            for param in ["Wavelength", "Brightness", "Voltage"]:
                                config = PRODUCTION_BINS[param]
                                bin_column = f"{param}_Bin"
                                counts = file_data[bin_column].value_counts().reindex(config['order'], fill_value=0)
                                for bin_code, count in counts.items():
                                    radar_data.append({
                                        '参数': config['name'],
                                        'Bin区': bin_code,
                                        '占比(%)': round((count / total_points * 100), 2)
                                    })

                            radar_df = pd.DataFrame(radar_data)

                            fig = px.line_polar(
                                radar_df,
                                r='占比(%)',
                                theta='Bin区',
                                color='参数',
                                line_close=True,
                                title=f'{file_name} 各参数Bin区分布雷达图'
                            )
                            fig.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100]))
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"生成综合分析时出错: {str(e)}")

                # 显示数据样本
                if st.session_state.production_data is not None and not st.session_state.production_data.empty:
                    st.subheader("产出分布数据样本")
                    display_columns = ['文件名', 'bin_code', '所属色区',
                                       'peak_wavelength1_nm', 'Wavelength_Bin',
                                       'LuminousFlux_lm', 'Brightness_Bin',
                                       'forward_voltage1_V', 'Voltage_Bin']
                    st.dataframe(st.session_state.production_data[display_columns].head(100))

            # 4. Mapping图分析选项卡
            with tab4:
                st.header("Mapping图分析")

                if len(st.session_state.dataframes) > 0:
                    # 优化布局
                    st.markdown("### Mapping图设置")

                    # 左侧控制面板，右侧图表
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        # 选择要分析的材料
                        material_file = st.selectbox(
                            "选择要分析的材料:",
                            list(st.session_state.dataframes.keys()),
                            key="mapping_material_selector"
                        )

                        # 产品类型选择
                        product_type_mapping = st.selectbox(
                            "产品类型",
                            ["NCSP", "CSP"],
                            index=0 if st.session_state.product_type == "NCSP" else 1,
                            key="mapping_product_type"
                        )

                        # 针对NCSP产品添加区域选择
                        if product_type_mapping == "NCSP":
                            ncsp_region = st.radio(
                                "选择显示区域",
                                ["全部区域", "区域1", "区域2"],
                                index=0 if st.session_state.ncsp_region is None else st.session_state.ncsp_region,
                                key="ncsp_region_selector"
                            )

                            if ncsp_region == "全部区域":
                                st.session_state.ncsp_region = None
                            elif ncsp_region == "区域1":
                                st.session_state.ncsp_region = 1
                            else:
                                st.session_state.ncsp_region = 2
                        else:
                            st.session_state.ncsp_region = None
                            st.info("CSP产品将使用pos_x和pos_y列，并自动归一化坐标从1开始")

                        # 色阶颜色选择
                        color_scale_name = st.selectbox(
                            "选择色阶方案",
                            list(color_scales.keys()),
                            key="color_scale_name_mapping",
                            format_func=lambda x: x
                        )
                        st.session_state.color_scale = color_scales[color_scale_name]

                        # 图表尺寸设置
                        st.subheader("图表尺寸")
                        map_width = st.slider("图表宽度 (像素)", 800, 2200, 2000, key="map_width")
                        map_height = st.slider("图表高度 (像素)", 500, 1500, 600, key="map_height")

                        # 异常点过滤
                        st.subheader("异常点过滤")
                        filter_outliers = st.checkbox("启用异常点过滤", True, key="filter_outliers")

                        if filter_outliers:
                            ciex_min = st.number_input("ciex 最小值", value=0.2, step=0.001, format="%.3f",
                                                       key="ciex_min")
                            ciex_max = st.number_input("ciex 最大值", value=0.4, step=0.001, format="%.3f",
                                                       key="ciex_max")
                            ciey_min = st.number_input("ciey 最小值", value=0.2, step=0.001, format="%.3f",
                                                       key="ciey_min")
                            ciey_max = st.number_input("ciey 最大值", value=0.4, step=0.001, format="%.3f",
                                                       key="ciey_max")

                            ciex_range = [ciex_min, ciex_max]
                            ciey_range = [ciey_min, ciey_max]
                        else:
                            ciex_range = None
                            ciey_range = None

                        # # 选择映射值
                        # value_column = st.selectbox(
                        #     "选择映射值",
                        #     ['ciex', 'ciey'],
                        #     key="value_column"
                        # )

                        # 颜色范围设置
                        st.subheader("颜色映射设置")
                        custom_color_range = st.checkbox("自定义颜色范围", False, key="custom_color_range")
                        if custom_color_range:
                            # 分别获取ciex和ciey的原始数据范围（未过滤bin_code）
                            material_df = st.session_state.dataframes[material_file]
                            ciex_min = float(material_df['ciex'].min())
                            ciex_max = float(material_df['ciex'].max())
                            ciey_min = float(material_df['ciey'].min())
                            ciey_max = float(material_df['ciey'].max())

                            st.markdown("#### CIE X 颜色范围")
                            col_ciex1, col_ciex2 = st.columns(2)
                            with col_ciex1:
                                ciex_color_min = st.number_input(
                                    "CIE X 最小值",
                                    value=ciex_min,
                                    step=0.0001,
                                    format="%.4f",
                                    key="ciex_color_min"
                                )
                            with col_ciex2:
                                ciex_color_max = st.number_input(
                                    "CIE X 最大值",
                                    value=ciex_max,
                                    step=0.0001,
                                    format="%.4f",
                                    key="ciex_color_max"
                                )

                            st.markdown("#### CIE Y 颜色范围")
                            col_ciey1, col_ciey2 = st.columns(2)
                            with col_ciey1:
                                ciey_color_min = st.number_input(
                                    "CIE Y 最小值",
                                    value=ciey_min,
                                    step=0.0001,
                                    format="%.4f",
                                    key="ciey_color_min"
                                )
                            with col_ciey2:
                                ciey_color_max = st.number_input(
                                    "CIE Y 最大值",
                                    value=ciey_max,
                                    step=0.0001,
                                    format="%.4f",
                                    key="ciey_color_max"
                                )

                            color_ranges = {
                                'ciex': [ciex_color_min, ciex_color_max],
                                'ciey': [ciey_color_min, ciey_color_max]
                            }
                        else:
                            color_ranges = {
                                'ciex': None,
                                'ciey': None
                            }

                        # 数据聚类与点大小优化
                        cluster_density = st.checkbox("根据密度调整点大小", False, key="cluster_density")

                        # 网格细分控制
                        show_grid_subdivisions = st.checkbox("显示细分网格", True, key="show_grid_subdivisions")

                        # 特殊标记设置
                        st.subheader("特殊标记设置")

                        # 创建一个容器来保存特殊标记设置
                        marker_container = st.container()

                        with marker_container:
                            add_marker = st.button("添加特殊标记", key="add_marker")

                            if st.button("重置所有标记", key="reset_markers"):
                                st.session_state.special_markers = []

                        if add_marker:
                            st.session_state.special_markers.append({
                                'condition': 'ciex > 0.3 and ciey > 0.325',
                                'color': '#FF0000',
                                'label': f'特殊点_{len(st.session_state.special_markers) + 1}'
                            })

                        # 显示已添加的特殊标记
                        if st.session_state.special_markers:
                            st.markdown("#### 当前特殊标记:")
                            for i, marker in enumerate(st.session_state.special_markers):
                                with st.expander(f"标记 {i + 1}: {marker['label']}"):
                                    col3, col4 = st.columns(2)
                                    with col3:
                                        condition = st.text_input(
                                            f"条件表达式",
                                            marker['condition'],
                                            key=f"condition_{i}",
                                            help="使用pandas查询语法，例如: 'ciex > 0.3 and ciey > 0.325'"
                                        )
                                    with col4:
                                        color = st.color_picker(
                                            f"标记颜色",
                                            marker['color'],
                                            key=f"marker_color_{i}"
                                        )

                                    label = st.text_input(
                                        f"标记标签",
                                        marker['label'],
                                        key=f"label_{i}"
                                    )

                                    col5, col6 = st.columns(2)
                                    with col5:
                                        if st.button(f"保存标记 {i + 1}", key=f"save_{i}"):
                                            st.session_state.special_markers[i] = {
                                                'condition': condition,
                                                'color': color,
                                                'label': label
                                            }
                                            st.success(f"标记 {i + 1} 已保存")
                                    with col6:
                                        if st.button(f"删除标记 {i + 1}", key=f"delete_{i}"):
                                            st.session_state.special_markers.pop(i)
                                            st.success(f"标记 {i + 1} 已删除")

                        # 添加单元格大小控制
                        st.subheader("单元格大小控制")
                        cell_size = st.slider(
                            "单元格大小比例",
                            min_value=0.5,
                            max_value=2.0,
                            value=st.session_state.get("cell_size", 0.9),
                            step=0.1,
                            key="cell_size_slider",
                            on_change=update_cell_size
                        )

                    with col2:
                        # 获取所选材料的数据
                        material_df = st.session_state.dataframes[material_file]
                        filtered_material_df = material_df[material_df['bin_code'].isin(selected_bin_codes)]

                        # 检查是否有数据
                        if filtered_material_df.empty:
                            st.warning("筛选后没有剩余数据点，请调整筛选条件")
                        else:
                            st.write(f"筛选后数据点数量: {len(filtered_material_df)}")

                        # 生成mapping图
                            if st.button("生成交互Mapping图", key="generate_map_button"):
                                # 先检查筛选后的数据是否为空
                                if filtered_material_df.empty:
                                    st.warning("无法生成Mapping图，可能是筛选后没有剩余数据点")
                                else:
                                    with st.spinner("正在生成Mapping图..."):
                                        # CIE X Mapping图（上）
                                        title_x = f'{material_file} - ciex Mapping图 ({product_type_mapping})'
                                        fig_x = generate_interactive_mapping_plot(
                                            filtered_material_df,
                                            'ciex',
                                            title_x,  # 传入标题，用于后续文件名
                                            map_width,
                                            map_height,
                                            filter_outliers,
                                            ciex_range,
                                            ciey_range,
                                            st.session_state.special_markers,
                                            st.session_state.color_scale,
                                            product_type_mapping,
                                            st.session_state.ncsp_region,
                                            st.session_state.cell_size,
                                            color_range=color_ranges['ciex'],
                                            cluster_density=cluster_density,
                                            show_grid_subdivisions=show_grid_subdivisions
                                        )
                                        st.plotly_chart(fig_x, use_container_width=True)

                                        # CIE Y Mapping图（下）
                                        title_y = f'{material_file} - ciey Mapping图 ({product_type_mapping})'
                                        fig_y = generate_interactive_mapping_plot(
                                            filtered_material_df,
                                            'ciey',
                                            title_y,  # 传入标题，用于后续文件名
                                            map_width,
                                            map_height,
                                            filter_outliers,
                                            ciex_range,
                                            ciey_range,
                                            st.session_state.special_markers,
                                            st.session_state.color_scale,
                                            product_type_mapping,
                                            st.session_state.ncsp_region,
                                            st.session_state.cell_size,
                                            color_range=color_ranges['ciey'],
                                            cluster_density=cluster_density,
                                            show_grid_subdivisions=show_grid_subdivisions
                                        )
                                        st.plotly_chart(fig_y, use_container_width=True)

                                with st.spinner("正在生成Mapping图..."):
                                    # CIE X Mapping图（上）
                                    title_x = f'{material_file} - ciex Mapping图 ({product_type_mapping})'
                                    fig_x = generate_interactive_mapping_plot(...)
                                    st.plotly_chart(fig_x, use_container_width=True)

                                    # CIE Y Mapping图（下）
                                    title_y = f'{material_file} - ciey Mapping图 ({product_type_mapping})'
                                    fig_y = generate_interactive_mapping_plot(...)
                                    st.plotly_chart(fig_y, use_container_width=True)

                else:
                    st.info("请上传材料文件以进行Mapping图分析")

    else:
        st.info("请上传分光数据文件开始分析")


if __name__ == "__main__":
    main()

