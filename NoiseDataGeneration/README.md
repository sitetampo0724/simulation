SNR值	含义	视觉效果
>20dB	信号非常清晰	几乎看不出噪声
10-20dB	信号清晰	能看到轻微噪声
0-10dB	信号和噪声相当	明显看到噪声
-10-0dB	噪声略大于信号	信号被噪声淹没
<-10dB	噪声远大于信号	几乎看不出原始信号

1. 信号基础参数（第 108-110 行）
Python
fs = 5000        # 采样率(Hz) — 越高波形越密集越"毛糙"
duration = 0.3   # 信号时长(秒)
表格
参数	           调大效果	                调小效果
fs	           波形更密集、噪声更细碎	    波形更稀疏、更平滑
duration	   更长的信号记录	            更短的信号记录

2. 噪声特征参数（generate_scope_noise 函数内）
毛刺噪声（第 24-30 行）
Python
glitch_prob = 0.15          # 毛刺出现概率 (0~1)
duration = np.random.randint(1, 5)   # 毛刺持续采样点数
amplitude = np.random.uniform(2, 6)  # 毛刺幅度
表格
参数	调大效果
glitch_prob	毛刺更密集
duration 范围	毛刺持续时间更长
amplitude 范围	毛刺更突出
突发噪声块（第 32-38 行）
Python
n_bursts = np.random.randint(2, 6)   # 突发块数量
length = np.random.randint(10, 50)   # 每个突发块长度
噪声成分权重（第 54-60 行）
Python
raw_noise = (white_noise * 0.35 +      # 白噪声权重
             glitches * 0.30 +          # 毛刺权重
             burst_noise * 0.15 +       # 突发块权重
             mains_noise * 0.10 +       # 工频干扰权重
             random_walk * 0.07 +       # 漂移权重
             quant_noise * 0.03)        # 量化噪声权重
调整权重比例可以改变噪声的"质感"——比如增大 glitches 权重会让波形更像您参考图中的那种密集毛刺感。
3. 工频干扰强度（第 41-43 行）
Python
mains_noise = (0.3 * np.sin(2 * np.pi * 50 * t) +    # 基波 50Hz
               0.15 * np.sin(2 * np.pi * 150 * t) +   # 3次谐波
               0.08 * np.sin(2 * np.pi * 250 * t))    # 5次谐波
如果您在中国（50Hz电网），这些数值不需要改。如果在日本/美国等60Hz地区，把 50 改成 60，谐波相应改成 180、300。
4. SNR 等级定义（第 113-119 行）
Python
'Very Clean':  {'range': (20, 30),  'count': 200, 'color': '#00FF00', 'noise_type': 'white'},
'Clean':       {'range': (10, 20),  'count': 200, 'color': '#66FF66', 'noise_type': 'mixed'},
'Moderate':    {'range': (0, 10),   'count': 200, 'color': '#FFFF00', 'noise_type': 'mixed'},
'Noisy':       {'range': (-10, 0),  'count': 200, 'color': '#FFAA00', 'noise_type': 'mixed'},
'Very Noisy':  {'range': (-20, -10),'count': 200, 'color': '#FF4444', 'noise_type': 'glitch'},
表格
配置项	说明
range	该等级的 SNR 范围 (dB)
count	该等级生成多少张图
color	示波器波形颜色（HTML色值）
noise_type	该等级使用哪种噪声模型
noise_type 可选值：
'white' — 纯高斯白噪声
'mixed' — 混合噪声（默认推荐）
'glitch' — 以毛刺噪声为主（最像您参考图的效果）
'burst' — 以突发块为主
'mains' — 以工频干扰为主
5. 信号参数范围（第 145-148 行）
Python
freqs = np.random.uniform(3, 25, n_samples)      # 频率范围 (Hz)
duties = np.random.uniform(15, 85, n_samples)    # 占空比范围 (%)
amps = np.random.uniform(0.5, 3.0, n_samples)    # 幅度范围 (V)
6. 绘图显示范围（第 211 行）
Python
ax.set_ylim(-5, 5)   # Y轴范围，决定波形在图中的上下边界

快速调出极致毛刺感：
glitch_prob = 0.25              # 毛刺更密集
amplitude = np.random.uniform(3, 8)   # 毛刺更高
# 所有等级 noise_type 都设为 'glitch'
# 权重: glitches 调高到 0.50