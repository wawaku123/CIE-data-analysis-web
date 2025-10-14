import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面
st.set_page_config(page_title="炉温曲线分析系统", page_icon="📊", layout="wide")

# 标题
st.title("🔥 无铅锡膏炉温曲线分析系统")
st.markdown("根据实测炉温曲线判断是否符合供应商推荐参数范围")

# 根据图片内容设置默认参数
DEFAULT_PARAMS = {
    't_smn': 150,  # 最低预热温度
    't_smax': 210,  # 最高预热温度
    't_l': 217,  # 液相线温度
    'tp_min': 235,  # 峰值温度最小值
    'tp_max': 255,  # 峰值温度最大值

    # 预热阶段参数
    'preheat_slope_min': 1.0,
    'preheat_slope_max': 2.5,
    'preheat_time_min': 30,
    'preheat_time_max': 60,

    # 恒温阶段参数
    'soak_slope_min': 0.5,
    'soak_slope_max': 1.0,
    'soak_time_min': 60,
    'soak_time_max': 90,

    # 熔融阶段参数
    'tl_to_tp_slope_min': 1.0,
    'tl_to_tp_slope_max': 2.0,
    'tl_time_min': 40,
    'tl_time_max': 70,

    # 冷却阶段参数
    'tp_to_tl_slope_min': 1.5,
    'tp_to_tl_slope_max': 3.0,
    'tp_time_max': 90
}

# 初始化session state
if 'saved_params' not in st.session_state:
    st.session_state.saved_params = DEFAULT_PARAMS.copy()

if 'data' not in st.session_state:
    st.session_state.data = None

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'calculation_details' not in st.session_state:
    st.session_state.calculation_details = {}

# 侧边栏 - 参数配置
st.sidebar.header("⚙️ 标准参数配置")

# 1. 基本温度参数
st.sidebar.subheader("基本温度参数")
t_smn = st.sidebar.number_input("恒温阶段最低预热温度 T_SMN (℃)",
                                value=st.session_state.saved_params.get('t_smn', 150),
                                key="t_smn")
t_smax = st.sidebar.number_input("恒温阶段最高预热温度 T_SMAX (℃)",
                                 value=st.session_state.saved_params.get('t_smax', 210),
                                 key="t_smax")
t_l = st.sidebar.number_input("锡膏液相线温度 TL (℃)",
                              value=st.session_state.saved_params.get('t_l', 217),
                              key="t_l")
tp_min = st.sidebar.number_input("峰值温度 TP 最小值 (℃)",
                                 value=st.session_state.saved_params.get('tp_min', 235),
                                 key="tp_min")
tp_max = st.sidebar.number_input("峰值温度 TP 最大值 (℃)",
                                 value=st.session_state.saved_params.get('tp_max', 255),
                                 key="tp_max")

# 2. 详细参数标准
st.sidebar.subheader("详细工艺参数标准")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown("**预热阶段**")
    preheat_slope_min = st.number_input("预热斜率最小值 (℃/s)",
                                        value=st.session_state.saved_params.get('preheat_slope_min', 1.0),
                                        key="preheat_slope_min")
    preheat_slope_max = st.number_input("预热斜率最大值 (℃/s)",
                                        value=st.session_state.saved_params.get('preheat_slope_max', 2.5),
                                        key="preheat_slope_max")
    preheat_time_min = st.number_input("预热时间最小值 (s)",
                                       value=st.session_state.saved_params.get('preheat_time_min', 30),
                                       key="preheat_time_min")
    preheat_time_max = st.number_input("预热时间最大值 (s)",
                                       value=st.session_state.saved_params.get('preheat_time_max', 60),
                                       key="preheat_time_max")

    st.markdown("**恒温阶段**")
    soak_slope_min = st.number_input("恒温斜率最小值 (℃/s)",
                                     value=st.session_state.saved_params.get('soak_slope_min', 0.5),
                                     key="soak_slope_min")
    soak_slope_max = st.number_input("恒温斜率最大值 (℃/s)",
                                     value=st.session_state.saved_params.get('soak_slope_max', 1.0),
                                     key="soak_slope_max")
    soak_time_min = st.number_input("恒温时间最小值 (s)",
                                    value=st.session_state.saved_params.get('soak_time_min', 60),
                                    key="soak_time_min")
    soak_time_max = st.number_input("恒温时间最大值 (s)",
                                    value=st.session_state.saved_params.get('soak_time_max', 90),
                                    key="soak_time_max")

with col2:
    st.markdown("**熔融阶段**")
    tl_to_tp_slope_min = st.number_input("TL到TP上升速率最小值 (℃/s)",
                                         value=st.session_state.saved_params.get('tl_to_tp_slope_min', 1.0),
                                         key="tl_to_tp_slope_min")
    tl_to_tp_slope_max = st.number_input("TL到TP上升速率最大值 (℃/s)",
                                         value=st.session_state.saved_params.get('tl_to_tp_slope_max', 2.0),
                                         key="tl_to_tp_slope_max")
    tl_time_min = st.number_input("TL以上时间最小值 (s)",
                                  value=st.session_state.saved_params.get('tl_time_min', 40),
                                  key="tl_time_min")
    tl_time_max = st.number_input("TL以上时间最大值 (s)",
                                  value=st.session_state.saved_params.get('tl_time_max', 70),
                                  key="tl_time_max")

    st.markdown("**冷却阶段**")
    tp_to_tl_slope_min = st.number_input("TP到TL下降速率最小值 (℃/s)",
                                         value=st.session_state.saved_params.get('tp_to_tl_slope_min', 1.5),
                                         key="tp_to_tl_slope_min")
    tp_to_tl_slope_max = st.number_input("TP到TL下降速率最大值 (℃/s)",
                                         value=st.session_state.saved_params.get('tp_to_tl_slope_max', 3.0),
                                         key="tp_to_tl_slope_max")
    tp_time_max = st.number_input("TP±5℃内时间最大值 (s)",
                                  value=st.session_state.saved_params.get('tp_time_max', 90),
                                  key="tp_time_max")

# 保存参数按钮
if st.sidebar.button("💾 保存当前参数配置"):
    st.session_state.saved_params = {
        't_smn': t_smn, 't_smax': t_smax, 't_l': t_l,
        'tp_min': tp_min, 'tp_max': tp_max,
        'preheat_slope_min': preheat_slope_min,
        'preheat_slope_max': preheat_slope_max,
        'preheat_time_min': preheat_time_min,
        'preheat_time_max': preheat_time_max,
        'soak_slope_min': soak_slope_min,
        'soak_slope_max': soak_slope_max,
        'soak_time_min': soak_time_min,
        'soak_time_max': soak_time_max,
        'tl_to_tp_slope_min': tl_to_tp_slope_min,
        'tl_to_tp_slope_max': tl_to_tp_slope_max,
        'tl_time_min': tl_time_min,
        'tl_time_max': tl_time_max,
        'tp_to_tl_slope_min': tp_to_tl_slope_min,
        'tp_to_tl_slope_max': tp_to_tl_slope_max,
        'tp_time_max': tp_time_max
    }
    st.sidebar.success("参数配置已保存！")

# 重置为默认参数按钮
if st.sidebar.button("🔄 重置为默认参数"):
    st.session_state.saved_params = DEFAULT_PARAMS.copy()
    st.sidebar.success("已重置为默认参数！")
    st.rerun()

# 主界面
tab1, tab2, tab3 = st.tabs(["📊 数据输入", "📈 曲线分析", "✅ 结果报告"])

with tab1:
    st.header("数据输入")

    # 数据输入方式选择
    input_method = st.radio("选择数据输入方式:",
                            ["使用示例数据", "粘贴数据", "上传文件"])

    data = None

    if input_method == "使用示例数据":
        # 提供示例数据
        example_data = """秒	TC1	TC2	TC3	TC4	TC5	TC6	TC7
0	73.73	28.51	28.4	29.62	29.84	29.56	29.34
2	82.28	29.01	28.9	30.28	30.4	30.06	29.78
4	96.4	29.62	29.56	31.01	31.06	30.67	30.23
6	104.34	30.23	30.23	31.84	31.84	31.56	30.9
8	112.67	31.12	31.17	33.28	32.51	32.45	31.56
10	114.06	32.9	32.62	35.9	33.95	33.78	32.73
12	113.95	36.06	37.34	40.4	37.01	36.62	35.28
14	113.56	39.78	41.9	43.84	40.56	40.73	38.62
16	113.56	43.28	46.51	47.23	44.28	44.84	42.17
18	112.78	46.62	50.06	50.56	47.62	48.56	45.56
20	112.12	50.06	53.84	53.34	51.01	52.01	48.95
22	110.73	53.01	57.12	56.34	53.95	55.01	51.95
24	110.12	55.78	59.62	59.06	56.51	57.84	54.78
26	112.78	58.62	62.51	61.4	58.9	60.28	57.28
28	114.23	61.01	65.01	63.95	61.23	62.67	59.73
30	152.01	63.01	67.62	65.78	63.4	65.01	61.9
32	164.95	64.73	68.95	67.62	65.23	66.78	63.78
34	171.23	67.56	71.78	72.28	67.67	68.9	66.12
36	173.28	71.56	76.51	76.12	71.67	72.78	69.78
38	172.95	75.56	81.23	80.51	75.78	77.28	73.62
40	173.12	79.4	84.73	84.23	79.56	81.62	77.56
42	173.12	83.12	89.06	87.67	83.45	85.78	81.51
44	172.34	86.51	92.62	91.62	86.78	89.45	85.12
46	172.23	90.06	95.95	94.51	90.01	92.84	88.56
48	171.51	93.17	99.56	97.9	92.95	96.12	92.01
50	173.51	96.28	102.4	101.12	95.9	99.12	95.01
52	190.62	99.06	105.73	104.01	98.9	102.12	98.06
54	221.45	101.28	107.95	106.12	101.56	104.84	100.73
56	228.45	103.51	109.06	108.56	103.4	106.56	102.73
58	227.4	107.01	114.17	113.78	106.56	109.28	105.67
60	228.78	111.28	118.23	118.34	110.9	113.45	109.62
62	229.01	115.73	123.84	123.12	115.51	118.06	113.9
64	228.84	120.12	127.95	127.51	119.84	122.73	118.4
66	227.62	124.51	132.28	131.62	124.17	127.45	122.9
68	225.9	128.67	136.62	135.95	128.28	131.62	127.17
70	225.34	132.78	140.06	139.9	132.01	135.51	131.12
72	217.23	136.9	144.34	143.45	135.84	139.34	135.12
74	207.51	140.4	148.17	147.17	139.34	143.01	138.67
76	197.95	143.23	151.28	149.78	142.78	146.56	141.9
78	194.4	145.51	152.9	151.28	145.17	149.12	144.23
80	192.9	147.56	154.23	153.12	146.78	150.56	146.17
82	191.73	149.67	155.51	154.45	148.23	151.95	147.78
84	190.4	151.4	156.9	155.9	149.78	153.12	149.28
86	189.78	153.01	157.78	157.01	150.95	154.28	150.78
88	189.45	154.56	159.17	158.12	152.12	155.34	152.12
90	188.12	156.01	160.28	159.51	153.23	156.34	153.28
92	187.62	157.28	161.12	160.45	154.4	157.34	154.56
94	186.62	158.51	162.28	161.45	155.45	158.28	155.56
96	188.95	159.73	163.17	162.51	156.51	159.28	156.78
98	190.12	160.67	164.06	163.23	157.45	160.12	157.78
100	191.73	161.28	164.51	163.67	158.28	161.06	158.62
102	202.78	162.01	164.73	164.06	158.78	161.4	159.23
104	205.84	162.73	164.95	164.67	159.12	161.73	159.62
106	205.78	163.73	166.34	166.23	160.01	162.45	160.51
108	205.78	165.06	167.62	167.51	161.4	163.56	161.78
110	205.12	166.45	169.28	169.01	162.78	164.9	163.06
112	205.06	167.67	170.4	170.23	164.17	166.23	164.45
114	204.62	169.06	171.73	171.34	165.56	167.67	165.84
116	203.78	170.28	172.95	172.67	166.9	168.95	167.23
118	203.73	171.56	174.12	173.9	168.06	170.17	168.51
120	202.56	172.9	175.4	175.01	169.45	171.23	169.84
122	200.78	174.01	176.4	176.06	170.45	172.4	170.95
124	207.84	174.78	177.45	177.06	171.51	173.45	171.95
126	210.67	175.67	178.01	177.73	172.45	174.45	172.9
128	211.84	176.56	178.84	178.78	173.28	175.06	173.78
130	212.23	177.73	180.06	179.84	174.45	176.01	174.73
132	211.9	178.9	181.23	181.12	175.62	177.23	175.95
134	211.56	180.01	182.12	182.06	176.78	178.28	177.06
136	211.56	181.17	183.17	183.17	178.01	179.56	178.23
138	211.4	182.28	184.23	184.23	179.12	180.56	179.51
140	211.06	183.45	185.23	185.12	180.17	181.73	180.67
142	210.45	184.4	186.23	186.17	181.17	182.73	181.73
144	210.12	185.28	187.12	187.06	182.01	183.56	182.67
146	214.01	186.23	188.12	187.9	183.01	184.62	183.62
148	222.51	186.9	188.73	188.51	183.9	185.45	184.45
150	225.73	187.62	189.01	189.06	184.4	185.95	185.06
152	225.67	188.62	190.4	190.45	185.34	186.78	186.01
154	226.01	189.78	191.56	191.67	186.62	187.9	187.12
156	225.95	190.95	192.9	193.01	188.01	189.28	188.23
158	226.17	192.23	194.12	194.34	189.28	190.45	189.62
160	225.56	193.51	195.34	195.45	190.67	191.9	190.95
162	225.45	194.67	196.51	196.73	191.9	193.06	192.23
164	226.34	195.78	197.51	197.78	193.01	194.28	193.4
166	230.06	196.9	198.67	198.95	194.28	195.56	194.67
168	229.67	198.06	199.9	200.12	195.34	196.67	195.56
170	256.34	199.06	200.9	201.06	196.45	197.73	196.73
172	268.12	200.01	201.73	202.06	197.51	198.84	197.73
174	271.06	201.56	203.9	204.56	198.9	200.12	198.95
176	272.23	203.51	206.34	207.01	201.23	202.23	201.06
178	271.67	205.67	209.06	209.78	203.9	204.78	203.28
180	271.62	208.06	211.28	212.17	206.4	207.45	205.67
182	270.78	210.56	213.78	214.4	209.12	210.01	208.06
184	269.45	212.67	216.12	216.95	211.51	212.45	210.45
186	268.17	214.95	218.28	218.9	213.67	214.73	212.73
188	263.56	217.12	220.51	221.01	215.73	216.9	214.95
190	253.4	219.12	222.45	222.9	217.67	218.95	216.84
192	247.01	220.73	224.51	224.56	219.62	220.84	218.78
194	242.67	222.06	225.51	225.45	221.01	222.45	220.28
196	235.67	223.01	226.01	225.73	221.67	223.4	221.17
198	233.45	223.67	226.23	226.01	222.06	224.01	221.9
200	232.4	224.12	226.23	226.12	222.17	224.01	222.23
202	231.9	224.51	226.28	226.01	222.17	223.9	222.45
204	231.4	224.67	226.28	226.17	222.17	224.01	222.78
206	231.06	224.84	226.17	226.01	222.23	224.01	222.9
208	230.73	225.01	226.17	226.01	222.23	224.06	223.17
210	230.62	225.12	226.12	226.01	222.23	224.01	223.23
212	230.17	225.17	226.01	225.78	222.23	223.9	223.45
214	229.23	225.28	226.01	226.01	222.23	223.9	223.51
216	228.12	225.34	225.95	225.9	222.34	223.95	223.56
218	226.73	225.4	226.01	225.95	222.28	223.84	223.56
220	226.06	225.4	225.78	225.73	222.34	223.84	223.62
222	225.73	225.23	225.56	225.56	222.23	223.73	223.56
224	225.56	225.12	225.28	225.45	222.17	223.62	223.56
226	225.45	224.95	225.17	225.28	222.17	223.62	223.56
228	225.56	224.9	225.01	225.23	222.17	223.62	223.56
230	225.56	224.9	224.84	225.01	222.12	223.45	223.45
232	225.62	224.73	224.73	224.84	222.01	223.23	223.45
234	226.12	224.62	224.56	224.67	222.01	223.17	223.4
236	226.4	224.62	224.34	224.62	221.9	223.01	223.28
238	230.45	224.51	224.34	224.62	222.01	223.01	223.28
240	256.01	224.51	224.17	224.73	222.17	223.12	223.45
242	265.67	224.51	224.17	224.78	222.23	223.23	223.4
244	271.34	224.78	224.62	225.95	222.56	223.45	223.62
246	272.95	225.95	226.78	227.56	224.01	224.51	224.62
248	273.78	227.17	228.84	229.56	225.67	226.23	226.01
250	273.17	228.73	230.56	231.28	227.67	228.06	227.45
252	272.4	230.4	232.67	233.34	229.45	230.01	229.17
254	271.62	232.06	234.28	235.12	231.4	231.9	230.9
256	271.01	233.56	235.95	236.67	233.12	233.56	232.45
258	269.9	235.01	237.56	238.12	234.62	235.12	234.01
260	264.06	236.56	239.06	239.51	236.17	236.78	235.56
262	258.4	237.67	240.56	240.78	237.67	238.34	236.9
264	251.78	238.67	241.51	241.56	238.84	239.56	238.12
266	245.12	239.4	241.84	241.84	239.56	240.45	239.06
268	239.95	239.78	241.84	241.84	239.73	240.78	239.4
270	236.62	240.01	241.84	241.67	239.62	240.78	239.56
272	232.06	239.84	241.51	241.34	239.45	240.56	239.51
274	230.34	239.78	241.28	240.95	239.01	240.28	239.23
276	229.17	239.56	240.84	240.51	238.56	239.84	239.06
278	226.4	239.23	240.45	240.06	238.12	239.56	238.84
280	210.62	238.95	240.17	239.73	237.73	239.23	238.4
282	202.34	238.62	239.56	239.17	237.17	238.67	238.06
284	193.23	238.12	239.06	238.45	236.56	238.12	237.51
286	180.62	237.73	238.62	238.01	236.01	237.51	236.95
288	130.28	237.17	238.12	237.28	235.4	236.95	236.23
290	96.84	236.17	237.12	235.67	234.78	236.28	235.78
292	81.01	234.56	235.95	232.78	233.34	235.23	235.01
294	69.95	232.28	233.4	229.4	231.06	233.23	233.73
296	69.45	229.95	229.73	226.4	228.28	230.34	231.9
298	66.28	226.95	225.78	221.73	225.01	227.17	229.17
300	63.56	221.84	219.28	214.34	220.23	222.23	225.12
302	67.95	215.95	213.23	207.67	213.78	215.28	219.73
304	64.51	209.78	204.51	201.45	207.78	208.78	213.62
306	68.78	204.4	198.67	196.06	201.51	202.23	207.67
308	70.06	198.28	191.56	190.17	195.73	196.28	201.62
310	66.17	192.28	184.95	184.9	189.95	190.28	195.9
312	54.9	186.62	179.73	178.51	184.4	184.45	190.06
314	59.62	180.67	173.51	174.06	179.23	178.78	184.51
316	58.95	175.62	168.84	168.06	174.23	173.45	179.34
318	50.45	170.51	164.12	162.45	169.17	168.51	174.23
320	55.17	165.06	158.95	157.4	163.73	163.06	169.23
322	49.23	159.34	152.12	152.28	158.95	158.12	163.95
324	50.56	154.51	147.84	147.67	154.23	153.12	159.17
326	50.95	148.95	142.12	142.95	149.34	148.17	154.23
328	55.78	144.01	136.95	138.9	144.95	143.4	149.45
330	56.45	139.45	132.78	133.67	140.45	138.73	144.78
332	51.67	134.56	128.23	130.12	136.45	134.51	140.4
334	53.12	130.62	124.4	125.73	132.62	130.34	136.34
336	45.45	127.01	120.62	121.4	128.73	126.73	132.28
338	49.84	122.95	117.23	117.67	124.67	123.06	128.4
340	45.62	118.67	112.95	114.28	121.28	119.67	124.56
342	46.4	115.34	109.28	110.73	118.23	116.45	121.01
344	45.9	111.62	105.73	108.12	114.9	113.12	117.56
346	42.78	108.01	102.23	104.9	111.9	110.12	114.06
348	48.06	104.78	99.56	101.56	108.62	106.73	111.01
350	47.73	101.45	96.34	98.95	105.73	103.9	107.67"""

        try:
            data = pd.read_csv(StringIO(example_data), sep='\t')
            st.session_state.data = data
            st.success("已加载示例数据！")
        except Exception as e:
            st.error(f"示例数据格式错误: {e}")

    elif input_method == "粘贴数据":
        pasted_data = st.text_area("粘贴炉温曲线数据 (制表符分隔):", height=300)
        if pasted_data:
            try:
                data = pd.read_csv(StringIO(pasted_data), sep='\t')
                st.session_state.data = data
                st.success(f"成功加载数据，共{len(data)}行")
            except Exception as e:
                st.error(f"数据格式错误: {e}")

    else:  # 上传文件
        uploaded_file = st.file_uploader("上传CSV或Excel文件", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                st.session_state.data = data
                st.success(f"成功加载数据，共{len(data)}行")
            except Exception as e:
                st.error(f"文件读取错误: {e}")

    # 显示数据预览
    if st.session_state.data is not None:
        st.subheader("数据预览")
        st.dataframe(st.session_state.data.head(10))

        # 检查必要的列
        required_cols = ['秒', 'TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7']
        if all(col in st.session_state.data.columns for col in required_cols):
            st.success("✅ 数据格式正确，可进行分析")
        else:
            st.error("数据缺少必要的列，请确保包含: 秒, TC2, TC3, TC4, TC5, TC6, TC7")

with tab2:
    st.header("炉温曲线分析")

    if st.session_state.data is None:
        st.warning("请先在'数据输入'标签页加载数据")
    else:
        data = st.session_state.data


        # 分析参数计算函数
        def analyze_temperature_curve(time_series, temp_series, analysis_params):
            results = {}

            # 基本参数
            max_temp = temp_series.max()
            max_time = time_series[temp_series.idxmax()]

            # 1. 预热阶段 (25℃到T_SMN)
            mask_preheat = (temp_series >= 25) & (temp_series <= analysis_params.get('t_smn', 150))
            preheat_times = time_series[mask_preheat]
            preheat_temps = temp_series[mask_preheat]
            if len(preheat_times) > 1:
                preheat_slope = (preheat_temps.iloc[-1] - preheat_temps.iloc[0]) / (
                            preheat_times.iloc[-1] - preheat_times.iloc[0])
                preheat_time = preheat_times.iloc[-1] - preheat_times.iloc[0]
                results['preheat_slope'] = preheat_slope
                results['preheat_time'] = preheat_time

            # 2. 恒温阶段 (T_SMN到T_SMAX)
            mask_soak = (temp_series >= analysis_params.get('t_smn', 150)) & (
                        temp_series <= analysis_params.get('t_smax', 210))
            soak_times = time_series[mask_soak]
            soak_temps = temp_series[mask_soak]
            if len(soak_times) > 1:
                soak_slope = (soak_temps.iloc[-1] - soak_temps.iloc[0]) / (soak_times.iloc[-1] - soak_times.iloc[0])
                soak_time = soak_times.iloc[-1] - soak_times.iloc[0]
                results['soak_slope'] = soak_slope
                results['soak_time'] = soak_time

            # 3. TL以上时间
            mask_above_tl = temp_series >= analysis_params.get('t_l', 217)
            if mask_above_tl.any():
                tl_times = time_series[mask_above_tl]
                if len(tl_times) > 0:
                    tl_time = tl_times.iloc[-1] - tl_times.iloc[0]
                    results['tl_time'] = tl_time

            # 4. TP±5℃内时间
            mask_near_tp = (temp_series >= (max_temp - 5)) & (temp_series <= (max_temp + 5))
            if mask_near_tp.any():
                tp_times = time_series[mask_near_tp]
                if len(tp_times) > 0:
                    tp_time = tp_times.iloc[-1] - tp_times.iloc[0]
                    results['tp_time'] = tp_time

            # 5. TL到TP上升速率
            mask_rising = (temp_series >= analysis_params.get('t_l', 217)) & (time_series <= max_time)
            if mask_rising.any():
                rising_times = time_series[mask_rising]
                rising_temps = temp_series[mask_rising]
                if len(rising_times) > 1:
                    rise_slope = (rising_temps.iloc[-1] - rising_temps.iloc[0]) / (
                                rising_times.iloc[-1] - rising_times.iloc[0])
                    results['tl_to_tp_slope'] = rise_slope

            # 6. TP到TL下降速率
            mask_falling = (temp_series >= analysis_params.get('t_l', 217)) & (time_series >= max_time)
            if mask_falling.any():
                falling_times = time_series[mask_falling]
                falling_temps = temp_series[mask_falling]
                if len(falling_times) > 1:
                    fall_slope = (falling_temps.iloc[-1] - falling_temps.iloc[0]) / (
                                falling_times.iloc[-1] - falling_times.iloc[0])
                    results['tp_to_tl_slope'] = abs(fall_slope)

            results['peak_temp'] = max_temp
            results['peak_time'] = max_time

            return results


        # 绘制曲线
        fig, ax = plt.subplots(figsize=(14, 8))

        # 定义区域颜色
        zone_colors = {
            'preheat': 'lightyellow',
            'soak': 'lightblue',
            'reflow': 'lightcoral',
            'cooling': 'lightgreen'
        }

        # 绘制标准区域 - 使用保存的参数
        ax.axhspan(25, st.session_state.saved_params.get('t_smn', 150), alpha=0.3, color=zone_colors['preheat'],
                   label='预热区')
        ax.axhspan(st.session_state.saved_params.get('t_smn', 150), st.session_state.saved_params.get('t_smax', 210),
                   alpha=0.3, color=zone_colors['soak'], label='浸润区')
        ax.axhspan(st.session_state.saved_params.get('t_l', 217), st.session_state.saved_params.get('tp_max', 255),
                   alpha=0.3, color=zone_colors['reflow'], label='熔融区')

        # 绘制关键温度线
        ax.axhline(y=st.session_state.saved_params.get('t_l', 217), color='red', linestyle='--', alpha=0.7,
                   label=f'液相线 TL ({st.session_state.saved_params.get("t_l", 217)}℃)')
        ax.axhline(y=st.session_state.saved_params.get('tp_min', 235), color='orange', linestyle='--', alpha=0.7,
                   label=f'TP范围')
        ax.axhline(y=st.session_state.saved_params.get('tp_max', 255), color='orange', linestyle='--', alpha=0.7)

        # 绘制各TC曲线
        tc_columns = ['TC2', 'TC3', 'TC4', 'TC5', 'TC6', 'TC7']
        colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink']

        all_results = {}

        for i, tc in enumerate(tc_columns):
            if tc in data.columns:
                ax.plot(data['秒'], data[tc], color=colors[i], label=tc, linewidth=2)

                # 分析该曲线
                results = analyze_temperature_curve(data['秒'], data[tc], st.session_state.saved_params)
                all_results[tc] = results

        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('温度 (℃)')
        ax.set_title('炉温曲线分析')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 300)

        st.pyplot(fig)

        # 保存分析结果
        st.session_state.analysis_results = all_results

with tab3:
    st.header("分析结果报告")

    if 'analysis_results' not in st.session_state:
        st.warning("请先在'曲线分析'标签页进行分析")
    else:
        all_results = st.session_state.analysis_results
        params = st.session_state.saved_params

        # 总体统计
        st.subheader("📈 总体统计")

        summary_data = []
        for tc, results in all_results.items():
            summary_data.append({
                '感温线': tc,
                '峰值温度(℃)': f"{results.get('peak_temp', 0):.1f}",
                '峰值时间(s)': f"{results.get('peak_time', 0):.1f}",
                'TL以上时间(s)': f"{results.get('tl_time', 0):.1f}",
                '预热斜率(℃/s)': f"{results.get('preheat_slope', 0):.2f}",
                '恒温时间(s)': f"{results.get('soak_time', 0):.1f}"
            })

        st.dataframe(pd.DataFrame(summary_data))

        # 详细分析
        st.subheader("🔍 详细分析结果")

        for tc, results in all_results.items():
            with st.expander(f"感温线 {tc} 分析结果"):
                col1, col2 = st.columns(2)

                with col1:
                    # 峰值温度检查
                    peak_temp = results.get('peak_temp', 0)
                    tp_min = st.session_state.saved_params.get('tp_min', 235)
                    tp_max = st.session_state.saved_params.get('tp_max', 255)
                    if tp_min <= peak_temp <= tp_max:
                        st.success(f"✅ 峰值温度: {peak_temp:.1f}℃ (符合 {tp_min}-{tp_max}℃)")
                    else:
                        st.error(f"❌ 峰值温度: {peak_temp:.1f}℃ (超出 {tp_min}-{tp_max}℃)")

                    # TL以上时间检查
                    tl_time = results.get('tl_time', 0)
                    tl_range = st.session_state.saved_params.get('tl_time', [40, 70])
                    if tl_range[0] <= tl_time <= tl_range[1]:
                        st.success(f"✅ TL以上时间: {tl_time:.1f}s (符合 {tl_range[0]}-{tl_range[1]}s)")
                    else:
                        st.error(f"❌ TL以上时间: {tl_time:.1f}s (超出 {tl_range[0]}-{tl_range[1]}s)")

                with col2:
                    # 预热斜率检查
                    preheat_slope = results.get('preheat_slope', 0)
                    preheat_range = st.session_state.saved_params.get('preheat_slope', [1.0, 2.5])
                    if preheat_range[0] <= preheat_slope <= preheat_range[1]:
                        st.success(
                            f"✅ 预热斜率: {preheat_slope:.2f}℃/s (符合 {preheat_range[0]}-{preheat_range[1]}℃/s)")
                    else:
                        st.error(f"❌ 预热斜率: {preheat_slope:.2f}℃/s (超出 {preheat_range[0]}-{preheat_range[1]}℃/s)")

                # 优化建议
                st.subheader("💡 优化建议")
                suggestions = []

                peak_temp = results.get('peak_temp', 0)
                if peak_temp < st.session_state.saved_params.get('tp_min', 235):
                    suggestions.append("提高峰值温度，确保达到锡膏熔融要求")
                elif peak_temp > st.session_state.saved_params.get('tp_max', 255):
                    suggestions.append("降低峰值温度，避免元件热损伤")

                tl_time = results.get('tl_time', 0)
                if tl_time < st.session_state.saved_params.get('tl_time', [40, 70])[0]:
                    suggestions.append("增加液相线以上时间，确保充分熔融")
                elif tl_time > st.session_state.saved_params.get('tl_time', [40, 70])[1]:
                    suggestions.append("减少液相线以上时间，避免过度氧化")

                preheat_slope = results.get('preheat_slope', 0)
                if preheat_slope < st.session_state.saved_params.get('preheat_slope', [1.0, 2.5])[0]:
                    suggestions.append("提高预热区升温速率")
                elif preheat_slope > st.session_state.saved_params.get('preheat_slope', [1.0, 2.5])[1]:
                    suggestions.append("降低预热区升温速率，避免热冲击")

                if suggestions:
                    for suggestion in suggestions:
                        st.info(suggestion)
                else:
                    st.success("✅ 曲线参数均在推荐范围内，工艺良好")

        # 导出报告
        st.subheader("📥 导出报告")
        if st.button("生成详细分析报告"):
            report_text = "无铅锡膏炉温曲线分析报告\n\n"
            report_text += f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_text += f"分析曲线数: {len(all_results)}\n\n"

            for tc, results in all_results.items():
                report_text += f"感温线 {tc}:\n"
                report_text += f"  峰值温度: {results.get('peak_temp', 0):.1f}℃\n"
                report_text += f"  TL以上时间: {results.get('tl_time', 0):.1f}s\n"
                report_text += f"  预热斜率: {results.get('preheat_slope', 0):.2f}℃/s\n\n"

            st.download_button(
                label="下载分析报告",
                data=report_text,
                file_name="炉温曲线分析报告.txt",
                mime="text/plain"
            )

# 页脚
st.markdown("---")
st.markdown("**炉温曲线分析系统 v1.0** - 基于无铅锡膏供应商推荐参数")
