import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import savemat

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FourierRecoApp:
    def __init__(self, root):
        self.root = root
        self.root.title('周期信号傅里叶重构')
        self.root.geometry('960x600')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        # -----------------------------
        # 核心数据
        # -----------------------------
        self.sig_waveform = None
        self.fourier_synthesis_result = None
        self.time_vector = None
        self.harmonic_data = None
        self.prev_table_data = {}
        self.current_mode = 'signal'   # signal / spectrum

        # -----------------------------
        # 参数变量
        # -----------------------------
        self.sig_type = tk.StringVar(value='选择信号')
        self.wave_count = tk.IntVar(value=2)
        self.amp = tk.DoubleVar(value=1.0)
        self.freq = tk.DoubleVar(value=1000.0)
        self.phase = tk.DoubleVar(value=0.0)
        self.sample_freq = tk.DoubleVar(value=20000.0)
        self.order = tk.IntVar(value=9)

        self.active_rows = 9
        self.selected_row = None

        self.build_ui()
        self.reset_table()
        self.update_info('✅ 系统初始化完成！')

    # =========================================================
    # UI
    # =========================================================
    def build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use('vista')
        except:
            pass

        style.configure('TLabel', font=('微软雅黑', 10))
        style.configure('TButton', font=('微软雅黑', 10))
        style.configure('TLabelframe.Label', font=('微软雅黑', 10, 'bold'))

        self.left = ttk.Frame(self.root)
        self.center = ttk.Frame(self.root)
        self.right = ttk.Frame(self.root)

        self.left.place(x=0, y=0, width=220, height=600)
        self.center.place(x=220, y=0, width=500, height=600)
        self.right.place(x=720, y=0, width=240, height=600)

        self.build_left()
        self.build_center()
        self.build_right()

    def build_left(self):
        # ================= 日志区 =================
        down = ttk.LabelFrame(self.left)
        down.place(x=0, y=0, width=220, height=110)

        ttk.Label(down, text='Info', font=('微软雅黑', 14, 'bold')).place(x=10, y=5)

        self.info_text = ScrolledText(down, font=('Consolas', 9))
        self.info_text.place(x=55, y=10, width=155, height=85)
        self.info_text.config(state='disabled')

        # ================= 按钮区 =================
        mid = ttk.LabelFrame(self.left)
        mid.place(x=0, y=110, width=220, height=140)

        ttk.Label(mid, text='阶数', font=('微软雅黑', 11)).place(x=10, y=12)
        tk.Entry(mid, textvariable=self.order, justify='center', font=('微软雅黑', 11), width=6)\
            .place(x=78, y=12)

        ttk.Button(mid, text='信号波形', command=self.show_signal).place(x=10, y=50, width=95, height=30)
        ttk.Button(mid, text='清除波形', command=self.clear_wave).place(x=115, y=50, width=95, height=30)
        ttk.Button(mid, text='傅里叶分解', command=self.decompose).place(x=10, y=90, width=95, height=30)
        ttk.Button(mid, text='傅里叶合成', command=self.synthesize).place(x=115, y=90, width=95, height=30)

        # ================= 参数区 =================
        up = ttk.LabelFrame(self.left)
        up.place(x=0, y=250, width=220, height=330)

        ttk.Label(up, text='周期信号', font=('微软雅黑', 16, 'bold')).place(x=10, y=10)

        ttk.Combobox(
            up,
            textvariable=self.sig_type,
            state='readonly',
            values=['选择信号', '方波', '三角波', '锯齿波', '抛物线'],
            font=('微软雅黑', 12),
            width=9
        ).place(x=100, y=12)

        ttk.Label(up, text='信号周期', font=('微软雅黑', 11)).place(x=10, y=52)
        tk.Entry(up, textvariable=self.wave_count, justify='center', font=('微软雅黑', 11), width=6)\
            .place(x=78, y=54)
        ttk.Button(up, text='默认值', command=self.reset_defaults).place(x=135, y=52, width=68, height=25)

        ttk.Label(up, text='信号参数', font=('微软雅黑', 16, 'bold')).place(x=10, y=92)

        ttk.Label(up, text='振幅', font=('微软雅黑', 11)).place(x=10, y=132)
        tk.Entry(up, textvariable=self.amp, justify='center', font=('微软雅黑', 11), width=10)\
            .place(x=78, y=132)
        ttk.Label(up, text='DIV', font=('微软雅黑', 11)).place(x=160, y=132)

        ttk.Label(up, text='频率', font=('微软雅黑', 11)).place(x=10, y=172)
        tk.Entry(up, textvariable=self.freq, justify='center', font=('微软雅黑', 11), width=10)\
            .place(x=78, y=172)
        ttk.Label(up, text='Hz', font=('微软雅黑', 11)).place(x=160, y=172)

        ttk.Label(up, text='相位', font=('微软雅黑', 11)).place(x=10, y=212)
        tk.Entry(up, textvariable=self.phase, justify='center', font=('微软雅黑', 11), width=10)\
            .place(x=78, y=212)
        ttk.Label(up, text='°', font=('微软雅黑', 11)).place(x=160, y=212)

        ttk.Label(up, text='采样频率', font=('微软雅黑', 11)).place(x=10, y=252)
        tk.Entry(up, textvariable=self.sample_freq, justify='center', font=('微软雅黑', 11), width=10)\
            .place(x=78, y=252)
        ttk.Label(up, text='Hz', font=('微软雅黑', 11)).place(x=160, y=252)

    def build_center(self):
        # 上图：原始信号 / 分解频谱
        fig1 = Figure(figsize=(4.8, 2.4), dpi=100)
        self.ax_wave = fig1.add_subplot(111)
        self.ax_wave.set_title('周期信号波形图')
        self.ax_wave.set_xlabel('时间 (s)')
        self.ax_wave.set_ylabel('幅值 (DIV)')

        self.canvas_wave = FigureCanvasTkAgg(fig1, master=self.center)
        self.canvas_wave.get_tk_widget().place(x=15, y=300, width=470, height=260)

        # 下图：合成信号
        fig2 = Figure(figsize=(4.8, 2.4), dpi=100)
        self.ax_syn = fig2.add_subplot(111)
        self.ax_syn.set_title('傅里叶合成波形')
        self.ax_syn.set_xlabel('时间 (s)')
        self.ax_syn.set_ylabel('幅值 (DIV)')

        self.canvas_syn = FigureCanvasTkAgg(fig2, master=self.center)
        self.canvas_syn.get_tk_widget().place(x=15, y=20, width=470, height=250)

    def build_right(self):
        # ================= 保存区 =================
        bottom = ttk.LabelFrame(self.right)
        bottom.place(x=0, y=0, width=240, height=110)

        ttk.Button(bottom, text='傅里叶合成数据', command=self.save_syn).place(x=50, y=18, width=140, height=32)
        ttk.Button(bottom, text='傅里叶分解数据', command=self.save_decomp).place(x=50, y=60, width=140, height=32)

        # ================= 谐波表 =================
        top = ttk.LabelFrame(self.right)
        top.place(x=0, y=110, width=240, height=470)

        ttk.Label(top, text='谐 波 数 值', font=('微软雅黑', 14, 'bold')).place(x=65, y=5)

        table_frame = tk.Frame(top, bd=1, relief='solid')
        table_frame.place(x=10, y=40, width=220, height=290)

        headers = ['CH', 'DIV', 'Hz', '°']
        widths = [4, 8, 8, 8]

        for c, h in enumerate(headers):
            tk.Label(
                table_frame,
                text=h,
                font=('微软雅黑', 10, 'bold'),
                relief='ridge',
                width=widths[c],
                bg='#f0f0f0'
            ).grid(row=0, column=c, sticky='nsew')

        self.table_vars = []
        self.table_entries = []

        for r in range(9):
            row_vars, row_entries = [], []
            for c in range(4):
                var = tk.StringVar()
                e = tk.Entry(table_frame, textvariable=var, justify='center',
                             font=('微软雅黑', 10), width=widths[c])
                e.grid(row=r + 1, column=c, sticky='nsew')

                if c == 0:
                    e.config(state='readonly', readonlybackground='white')
                else:
                    e.bind('<FocusIn>', lambda ev, rr=r, cc=c: self.store_prev(rr, cc))
                    e.bind('<FocusOut>', lambda ev, rr=r, cc=c: self.validate_cell(rr, cc))

                e.bind('<Button-1>', lambda ev, rr=r: self.select_row(rr))

                row_vars.append(var)
                row_entries.append(e)

            self.table_vars.append(row_vars)
            self.table_entries.append(row_entries)

        ttk.Button(top, text='增加一阶谐波', command=self.add_row).place(x=15, y=350, width=95, height=28)
        ttk.Button(top, text='删除一阶谐波', command=self.delete_row).place(x=15, y=385, width=95, height=28)
        ttk.Button(top, text='数值初始化', command=self.reset_table).place(x=125, y=350, width=85, height=28)

    # =========================================================
    # 通用
    # =========================================================
    def timestamped(self, msg):
        return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

    def update_info(self, msg):
        old_text = self.info_text.get('1.0', 'end').strip()
        old = old_text.split('\n') if old_text else []
        lines = [self.timestamped(msg)] + old
        lines = lines[:4]

        self.info_text.config(state='normal')
        self.info_text.delete('1.0', 'end')
        self.info_text.insert('1.0', '\n'.join(lines))
        self.info_text.config(state='disabled')

    def on_close(self):
        if messagebox.askyesno('退出程序', '确定要关闭程序吗？'):
            self.root.destroy()

    def reset_defaults(self):
        self.sig_type.set('选择信号')
        self.wave_count.set(2)
        self.amp.set(1.0)
        self.freq.set(1000.0)
        self.phase.set(0.0)
        self.sample_freq.set(20000.0)
        self.order.set(9)
        self.update_info('✅ 参数已恢复默认值')

    def validate_basic_params(self):
        if self.sig_type.get() == '选择信号':
            raise ValueError('请选择周期信号类型！')
        if self.wave_count.get() <= 0:
            raise ValueError('信号周期数必须大于 0！')
        if self.freq.get() <= 0:
            raise ValueError('信号频率必须大于 0！')
        if self.sample_freq.get() <= 0:
            raise ValueError('采样频率必须大于 0！')
        if self.order.get() <= 0:
            raise ValueError('谐波阶数必须大于 0！')
        if self.order.get() > 9:
            raise ValueError('当前界面最多支持 9 阶谐波显示！')
        if self.sample_freq.get() < 2 * self.freq.get():
            raise ValueError('采样频率应至少满足奈奎斯特采样条件：Fs >= 2*f！')

    def get_time_axis_for_plot(self, t):
        xmax = t[-1] if len(t) > 0 else 0
        if xmax < 0.1:
            return t * 1000, xmax * 1000, '时间 (ms)'
        return t, xmax, '时间 (s)'

    # =========================================================
    # 谐波表
    # =========================================================
    def reset_table(self):
        self.active_rows = 9
        self.selected_row = None

        for r in range(9):
            self.table_vars[r][0].set(str(r + 1))
            for c in range(1, 4):
                self.table_vars[r][c].set('')

        self.harmonic_data = np.full((9, 4), np.nan)
        self.harmonic_data[:, 0] = np.arange(1, 10)
        self.update_info('✅ 谐波表已初始化')

    def store_prev(self, r, c):
        self.prev_table_data[(r, c)] = self.table_vars[r][c].get()

    def validate_cell(self, r, c):
        if c == 0:
            return

        value = self.table_vars[r][c].get().strip()
        if value == '':
            self.update_harmonic_storage()
            return

        try:
            float(value)
            self.update_harmonic_storage()
        except:
            messagebox.showerror('输入错误', '请输入有效数字或留空！')
            self.table_vars[r][c].set(self.prev_table_data.get((r, c), ''))

    def update_harmonic_storage(self):
        data = []
        for r in range(self.active_rows):
            row = [float(r + 1)]
            for c in range(1, 4):
                txt = self.table_vars[r][c].get().strip()
                row.append(np.nan if txt == '' else float(txt))
            data.append(row)
        self.harmonic_data = np.array(data, dtype=float)

    def current_table_data(self):
        self.update_harmonic_storage()
        return self.harmonic_data.copy()

    def select_row(self, r):
        self.selected_row = r
        for i, row in enumerate(self.table_entries):
            for e in row:
                color = '#dbeafe' if i == r else 'white'
                try:
                    if e.cget('state') == 'readonly':
                        e.config(readonlybackground=color)
                    else:
                        e.config(bg=color)
                except:
                    pass

    def add_row(self):
        limit = min(self.order.get(), 9)
        if self.active_rows >= limit:
            messagebox.showinfo('提示', '已达到最大阶数！')
            self.update_info('⚠️ 已达到最大阶数')
            return

        r = self.active_rows
        self.table_vars[r][0].set(str(r + 1))
        for c in range(1, 4):
            self.table_vars[r][c].set('')

        self.active_rows += 1
        self.update_harmonic_storage()
        self.update_info('✅ 已增加一阶谐波')

    def delete_row(self):
        if self.selected_row is None or self.selected_row >= self.active_rows:
            messagebox.showinfo('提示', '请先选择要删除的行')
            return

        for r in range(self.selected_row, self.active_rows - 1):
            for c in range(1, 4):
                self.table_vars[r][c].set(self.table_vars[r + 1][c].get())

        last = self.active_rows - 1
        for c in range(1, 4):
            self.table_vars[last][c].set('')

        self.active_rows = max(1, self.active_rows - 1)
        self.selected_row = None
        self.update_harmonic_storage()
        self.update_info('✅ 已删除选定谐波')

    # =========================================================
    # 信号生成
    # =========================================================
    def generate_signal(self):
        self.validate_basic_params()

        sig_type = self.sig_type.get()
        amp = self.amp.get()
        freq = self.freq.get()
        phase_deg = self.phase.get()
        fs = self.sample_freq.get()
        count = self.wave_count.get()

        T = 1 / freq
        total_time = count * T
        n = int(total_time * fs)
        n = max(n, 200)

        t = np.linspace(0, total_time, n, endpoint=False)
        ang = 2 * np.pi * freq * t + np.deg2rad(phase_deg)

        if sig_type == '方波':
            y = amp * signal.square(ang)

        elif sig_type == '三角波':
            y = amp * signal.sawtooth(ang, width=0.5)

        elif sig_type == '锯齿波':
            y = amp * signal.sawtooth(ang, width=1.0)

        elif sig_type == '抛物线':
            x = ((freq * t + phase_deg / 360.0) % 1.0)
            y = amp * (1 - 4 * (x - 0.5) ** 2)

        else:
            y = np.zeros_like(t)

        return t, y

    # =========================================================
    # 功能：显示原始信号
    # =========================================================
    def show_signal(self):
        try:
            t, y = self.generate_signal()
            self.time_vector = t
            self.sig_waveform = y
            self.current_mode = 'signal'

            t_plot, xmax_plot, xlabel = self.get_time_axis_for_plot(t)

            self.ax_wave.clear()
            self.ax_wave.plot(t_plot, y, color='b', linewidth=1)
            self.ax_wave.set_xlim(0, xmax_plot)
            y_lim = max(abs(np.min(y)), abs(np.max(y)), 1e-6)
            self.ax_wave.set_ylim(-1.2 * y_lim, 1.2 * y_lim)
            self.ax_wave.set_xlabel(xlabel)
            self.ax_wave.set_ylabel('幅值 (DIV)')
            self.ax_wave.set_title('周期信号波形图')
            self.ax_wave.grid(True)
            self.canvas_wave.draw()

            self.update_info('✅ 信号波形已输出')
        except Exception as e:
            messagebox.showerror('错误', str(e))

    # =========================================================
    # 功能：傅里叶分解
    # =========================================================
    def decompose(self):
        try:
            t, y = self.generate_signal()
            self.time_vector = t
            self.sig_waveform = y
            self.current_mode = 'spectrum'

            fs = self.sample_freq.get()
            base_freq = self.freq.get()
            order = self.order.get()

            n = len(y)
            fft_result = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(n, d=1 / fs)

            amps = 2 * np.abs(fft_result) / n
            phases = np.angle(fft_result, deg=True)

            harmonic_freqs = []
            harmonic_amps = []
            harmonic_phases = []

            for k in range(1, order + 1):
                target_freq = k * base_freq
                idx = np.argmin(np.abs(freqs - target_freq))

                harmonic_freqs.append(freqs[idx])
                harmonic_amps.append(amps[idx])
                harmonic_phases.append(phases[idx])

            harmonic_freqs = np.array(harmonic_freqs)
            harmonic_amps = np.array(harmonic_amps)
            harmonic_phases = np.array(harmonic_phases)

            self.harmonic_data = np.zeros((order, 4))
            self.harmonic_data[:, 0] = np.arange(1, order + 1)
            self.harmonic_data[:, 1] = harmonic_amps
            self.harmonic_data[:, 2] = harmonic_freqs
            self.harmonic_data[:, 3] = harmonic_phases

            self.active_rows = order
            for r in range(9):
                self.table_vars[r][0].set(str(r + 1))
                if r < order:
                    self.table_vars[r][1].set(f'{self.harmonic_data[r, 1]:.6f}'.rstrip('0').rstrip('.'))
                    self.table_vars[r][2].set(f'{self.harmonic_data[r, 2]:.6f}'.rstrip('0').rstrip('.'))
                    self.table_vars[r][3].set(f'{self.harmonic_data[r, 3]:.6f}'.rstrip('0').rstrip('.'))
                else:
                    for c in range(1, 4):
                        self.table_vars[r][c].set('')

            self.ax_wave.clear()
            ax2 = self.ax_wave.twinx()

            self.ax_wave.stem(harmonic_freqs, harmonic_amps, linefmt='b-', markerfmt='bo', basefmt=' ')
            ax2.stem(harmonic_freqs, harmonic_phases, linefmt='r-', markerfmt='rs', basefmt=' ')

            self.ax_wave.set_xlabel('频率 (Hz)')
            self.ax_wave.set_ylabel('幅值 (DIV)', color='b')
            ax2.set_ylabel('相位 (°)', color='r')
            self.ax_wave.set_title('傅里叶幅值谱与相位谱')
            self.ax_wave.grid(True)

            if len(harmonic_freqs) > 0:
                self.ax_wave.set_xlim(0, harmonic_freqs[-1] + base_freq / 2)

            self.canvas_wave.draw()
            self.update_info('✅ 傅里叶频谱图已输出')
        except Exception as e:
            messagebox.showerror('错误', str(e))

    # =========================================================
    # 功能：傅里叶合成
    # =========================================================
    def synthesize(self):
        try:
            self.validate_basic_params()

            table_data = self.current_table_data()
            if len(table_data) == 0:
                messagebox.showerror('错误', '谐波数据为空！')
                return

            freq = self.freq.get()
            count = self.wave_count.get()
            fs = self.sample_freq.get()

            total_time = count / freq
            n = int(total_time * fs)
            n = max(n, 200)

            t = np.linspace(0, total_time, n, endpoint=False)
            y = np.zeros_like(t)

            for row in table_data:
                amp_i = row[1]
                freq_i = row[2]
                phase_i = row[3]

                if np.isnan(amp_i) or np.isnan(freq_i) or np.isnan(phase_i):
                    continue

                y += amp_i * np.sin(2 * np.pi * freq_i * t + np.deg2rad(phase_i))

            self.fourier_synthesis_result = y
            self.time_vector = t

            t_plot, xmax_plot, xlabel = self.get_time_axis_for_plot(t)

            self.ax_syn.clear()
            self.ax_syn.plot(t_plot, y, color='purple', linewidth=1)
            self.ax_syn.set_xlim(0, xmax_plot)
            y_lim = max(abs(np.min(y)), abs(np.max(y)), 1e-6)
            self.ax_syn.set_ylim(-1.2 * y_lim, 1.2 * y_lim)
            self.ax_syn.set_xlabel(xlabel)
            self.ax_syn.set_ylabel('幅值 (DIV)')
            self.ax_syn.set_title('傅里叶合成波形')
            self.ax_syn.grid(True)
            self.canvas_syn.draw()

            self.update_info('✅ 傅里叶合成波形已输出')
        except Exception as e:
            messagebox.showerror('错误', str(e))

    # =========================================================
    # 清除图像
    # =========================================================
    def clear_wave(self):
        self.ax_wave.clear()
        self.ax_wave.set_title('周期信号波形图')
        self.ax_wave.set_xlabel('时间 (s)')
        self.ax_wave.set_ylabel('幅值 (DIV)')
        self.canvas_wave.draw()

        self.ax_syn.clear()
        self.ax_syn.set_title('傅里叶合成波形')
        self.ax_syn.set_xlabel('时间 (s)')
        self.ax_syn.set_ylabel('幅值 (DIV)')
        self.canvas_syn.draw()

        self.sig_waveform = None
        self.fourier_synthesis_result = None
        self.time_vector = None
        self.current_mode = 'signal'

        self.update_info('✅ 图像已清除')

    # =========================================================
    # 保存
    # =========================================================
    def save_path(self, default_name):
        return filedialog.asksaveasfilename(
            title='保存数据',
            defaultextension='.xlsx',
            initialfile=default_name,
            filetypes=[
                ('Excel 文件', '*.xlsx'),
                ('CSV 文件', '*.csv'),
                ('MAT 文件', '*.mat')
            ]
        )

    def save_common(self, path, df, params, harmonics):
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == '.xlsx':
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                    pd.DataFrame([params]).to_excel(writer, sheet_name='Parameters', index=False)
                    pd.DataFrame(harmonics, columns=['CH', 'DIV', 'Hz', 'Phase']).to_excel(
                        writer, sheet_name='Harmonics', index=False
                    )

            elif ext == '.csv':
                df.to_csv(path, index=False)
                base = os.path.splitext(path)[0]
                pd.DataFrame([params]).to_csv(base + '_params.csv', index=False)
                pd.DataFrame(harmonics, columns=['CH', 'DIV', 'Hz', 'Phase']).to_csv(
                    base + '_harmonics.csv', index=False
                )

            elif ext == '.mat':
                savemat(path, {
                    'time': np.array(df.iloc[:, 0]),
                    'data': np.array(df.iloc[:, 1]),
                    'harmonics': harmonics,
                    'params': params
                })

            else:
                messagebox.showerror('错误', '不支持的文件格式！')
                return

            self.update_info(f'✅ 数据已保存: {os.path.basename(path)}')
        except Exception as e:
            self.update_info(f'❌ 保存失败: {e}')
            messagebox.showerror('保存失败', str(e))

    def save_decomp(self):
        if self.sig_waveform is None or self.time_vector is None:
            messagebox.showerror('错误', '请先生成信号或进行傅里叶分解后再保存！')
            return

        df = pd.DataFrame({
            'Time': self.time_vector,
            'WaveData': self.sig_waveform
        })

        params = {
            'SignalType': self.sig_type.get(),
            'Amplitude': self.amp.get(),
            'Frequency': self.freq.get(),
            'Phase': self.phase.get(),
            'SampleFrequency': self.sample_freq.get(),
            'WaveCount': self.wave_count.get(),
            'Order': self.order.get()
        }

        path = self.save_path(f'FourierDecompositionData_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        self.save_common(path, df, params, self.current_table_data())

    def save_syn(self):
        if self.fourier_synthesis_result is None or self.time_vector is None:
            messagebox.showerror('错误', '请先生成傅里叶合成波形后再保存！')
            return

        df = pd.DataFrame({
            'Time': self.time_vector,
            'FourierSynthesisData': self.fourier_synthesis_result
        })

        params = {
            'SignalType': self.sig_type.get(),
            'Amplitude': self.amp.get(),
            'Frequency': self.freq.get(),
            'Phase': self.phase.get(),
            'SampleFrequency': self.sample_freq.get(),
            'WaveCount': self.wave_count.get(),
            'Order': self.order.get()
        }

        path = self.save_path(f'FourierSynthesisData_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        self.save_common(path, df, params, self.current_table_data())


if __name__ == '__main__':
    root = tk.Tk()
    app = FourierRecoApp(root)
    root.mainloop()