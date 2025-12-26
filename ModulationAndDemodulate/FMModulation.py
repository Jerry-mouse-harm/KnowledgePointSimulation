"""
FM调频调制解调器
调制信号: s(t) = A_c * cos(2πf_c*t + 2π*k_f*∫m(τ)dτ)
其中 k_f 是频率偏移常数
"""
import numpy as np
import sympy as sp
from sympy import sin, cos, exp, pi, lambdify, latex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from BasicFilter.BasicFilter import BasicFilter

t = sp.Symbol('t', real=True)


class FMModulation(BasicFilter):
    """
    FM调频调制解调器
    调制信号: s(t) = A_c * cos(2πf_c*t + 2π*k_f*∫m(τ)dτ)
    其中 k_f 是频率偏移常数
    """
    def __init__(self, sym_exper, carrier_freq, sample_frequency, T, freq_deviation=50):
        """
        参数:
            sym_exper: sympy表达式，基带信号
            carrier_freq: 载波频率 (Hz)
            sample_frequency: 采样频率 (Hz)
            T: 信号时长 (秒)
            freq_deviation: 最大频偏 (Hz)
        """
        super().__init__(sym_exper, sample_frequency, T)
        self.carrier_freq = carrier_freq
        self.freq_deviation = freq_deviation

        # 归一化基带信号
        self.baseband_signal = self.func
        self.normalized_baseband = self.normalize_signal(self.baseband_signal)

        # 调制和解调信号
        self.modulated_signal = None
        self.demodulated_signal = None

    def normalize_signal(self, signal):
        """归一化信号"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal

    def modulate(self):
        """
        FM调制
        瞬时频率: f(t) = f_c + k_f * m(t)
        瞬时相位: φ(t) = 2π∫f(t)dt = 2πf_c*t + 2πk_f*∫m(t)dt
        """
        # 计算基带信号的积分（使用累积求和近似）
        dt = self.T / self.N
        integral_m = np.cumsum(self.normalized_baseband) * dt

        # 瞬时相位
        phase = 2 * np.pi * self.carrier_freq * self.t_vals + \
                2 * np.pi * self.freq_deviation * integral_m

        # FM调制信号
        self.modulated_signal = np.cos(phase)

        # 计算调制信号的FFT
        self.modulated_fft = np.fft.fft(self.modulated_signal)
        self.modulated_amplitude = np.abs(self.modulated_fft)

        return self.modulated_signal

    def demodulate(self):
        """
        FM解调 - 使用微分检波法
        基本思想：瞬时频率 = d(φ(t))/dt
        """
        if self.modulated_signal is None:
            self.modulate()

        # 使用希尔伯特变换获取解析信号
        from scipy.signal import hilbert
        analytic_signal = hilbert(self.modulated_signal)

        # 计算瞬时相位
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # 计算瞬时频率（相位的微分）
        dt = self.T / self.N
        instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi * dt)

        # 补齐长度
        instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])

        # 去除载波频率，得到基带信号
        self.demodulated_signal = instantaneous_freq - self.carrier_freq

        # 归一化到合理范围
        self.demodulated_signal = self.demodulated_signal / self.freq_deviation

        # 计算解调信号的FFT
        self.demodulated_fft = np.fft.fft(self.demodulated_signal)
        self.demodulated_amplitude = np.abs(self.demodulated_fft)

        return self.demodulated_signal

    def visualize(self, carrier_freq_range = None):
        """可视化调制解调过程"""
        # 执行调制和解调
        if carrier_freq_range is None:
            print('carrier_freq_range 未设置')
            exit(-1)

        num_steps = 50
        min_carrier_freq, max_carrier_freq = carrier_freq_range
        cur_carrier_freq = np.linspace(min_carrier_freq, max_carrier_freq, num_steps)

        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'基带信号: {sp.pretty(self.sym_exper)}',
                '基带信号频谱',
                f'FM调制信号 (fc={self.carrier_freq}Hz, Δf={self.freq_deviation}Hz)',
                '调制信号频谱',
                'FM解调信号',
                '解调信号频谱'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 为每一个载波频率生成图像
        frames = []

        # 首先为第一个频率添加初始数据
        self.carrier_freq = cur_carrier_freq[0]
        self.modulate()
        self.demodulate()

        # 1. 基带信号
        fig.add_trace(
            go.Scatter(x=self.t_vals, y=self.normalized_baseband,
                                mode='lines', name='基带信号',
                                line=dict(color='blue', width=1.5)), row=1, col=1)

        # 2. 基带信号频谱
        fig.add_trace(
            go.Scatter(x=self.fft_freq, y=self.amplitude/self.N,
                                mode='lines', name='基带频谱',
                                line=dict(color='blue', width=1)), row=1, col=2)

        # 3. 调制信号
        fig.add_trace(
            go.Scatter(x=self.t_vals, y=self.modulated_signal,
                                mode='lines', name='调制信号',
                                line=dict(color='red', width=1)), row=2, col=1)

        # 4. 调制信号频谱
        fig.add_trace(
            go.Scatter(x=self.fft_freq, y=self.modulated_amplitude/self.N,
                                mode='lines', name='调制频谱',
                                line=dict(color='red', width=1)), row=2, col=2)

        # 5. 解调信号
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.demodulated_signal,

                                mode='lines', name='解调信号',
                                line=dict(color='green', width=1.5)), row=3, col=1)

        # 6. 解调信号频谱
        fig.add_trace(go.Scatter(x=self.fft_freq, y=self.demodulated_amplitude/self.N,
                                mode='lines', name='解调频谱',
                                line=dict(color='green', width=1)), row=3, col=2)

        # 为每个载波频率创建frames
        for i, cur_freq in enumerate(cur_carrier_freq):
            self.carrier_freq = cur_freq
            print(f'生成帧 {i + 1}/{num_steps}: carrier_freq = {cur_freq:.2f} Hz')
            self.modulate()
            self.demodulate()

            # 创建frame数据 - 注意：索引0-5对应6个subplot
            frames_data = [
                # trace 0: 基带信号 (row=1, col=1)
                go.Scatter(x=self.t_vals, y=self.normalized_baseband,
                           mode='lines', line=dict(color='blue', width=1.5)),

                # trace 1: 基带频谱 (row=1, col=2)
                go.Scatter(x=self.fft_freq, y=self.amplitude / self.N,
                           mode='lines', line=dict(color='blue', width=1)),

                # trace 2: 调制信号 (row=2, col=1)
                go.Scatter(x=self.t_vals, y=self.modulated_signal,
                           mode='lines', line=dict(color='red', width=1)),

                # trace 3: 调制频谱 (row=2, col=2)
                go.Scatter(x=self.fft_freq, y=self.modulated_amplitude / self.N,
                           mode='lines', line=dict(color='red', width=1)),

                # trace 4: 解调信号 (row=3, col=1)
                go.Scatter(x=self.t_vals, y=self.demodulated_signal,
                           mode='lines', line=dict(color='green', width=1.5)),

                # trace 5: 解调频谱 (row=3, col=2)
                go.Scatter(x=self.fft_freq, y=self.demodulated_amplitude / self.N,
                           mode='lines', line=dict(color='green', width=1))
            ]

            # 创建frame - 修复name命名
            frames.append(go.Frame(
                data=frames_data,
                name=str(i),  # ← 关键修复：使用简单的索引作为name
                layout=go.Layout(
                    title_text=f'FM调制解调系统 - 载波频率: {cur_freq:.2f} Hz'
                )
            ))

            # 添加frames到figure
            fig.frames = frames

            # 创建slider - 修复args引用
            sliders = [dict(
                active=0,
                yanchor="top",
                y=-0.05,
                xanchor="left",
                currentvalue=dict(
                    prefix="载波频率: ",
                    suffix=" Hz",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                steps=[dict(
                    args=[[str(i)],  # ← 关键修复：与frame的name一致
                          dict(frame=dict(duration=0, redraw=True),
                               mode="immediate",
                               transition=dict(duration=0))],
                    method="animate",
                    label=f'{freq:.1f}'
                ) for i, freq in enumerate(cur_carrier_freq)]
            )]
            # 添加播放/暂停按钮
            updatemenus = [dict(
                type="buttons",
                showactive=False,
                y=-0.05,
                x=0.05,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="播放",
                         method="animate",
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                          fromcurrent=True,
                                          mode="immediate",
                                          transition=dict(duration=0))]),
                    dict(label="暂停",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate",
                                            transition=dict(duration=0))])
                ]
            )]

            # 更新坐标轴
            fig.update_xaxes(title_text="时间 (s)", row=1, col=1)
            fig.update_xaxes(title_text="频率 (Hz)", row=1, col=2)
            fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
            fig.update_xaxes(title_text="频率 (Hz)", row=2, col=2)
            fig.update_xaxes(title_text="时间 (s)", row=3, col=1)
            fig.update_xaxes(title_text="频率 (Hz)", row=3, col=2)

            fig.update_yaxes(title_text="幅度", row=1, col=1)
            fig.update_yaxes(title_text="幅度", row=1, col=2)
            fig.update_yaxes(title_text="幅度", row=2, col=1)
            fig.update_yaxes(title_text="幅度", row=2, col=2)
            fig.update_yaxes(title_text="幅度", row=3, col=1)
            fig.update_yaxes(title_text="幅度", row=3, col=2)

            # 更新布局
            fig.update_layout(
                title=dict(
                    text=f'AM调制解调系统 - 信号: x(t) = {sp.pretty(self.sym_exper)}',
                    x=0.5,
                    xanchor='center'
                ),
                sliders=sliders,
                updatemenus=updatemenus,
                height=900,
                width=1800,
                font=dict(size=10),
                showlegend=True,
                hovermode='x unified'
            )

            self.interactive_fig = fig

        return fig
        """
        # 更新坐标轴
        for i in range(1, 4):
            fig.update_xaxes(title_text="时间 (s)", row=i, col=1)
            fig.update_xaxes(title_text="频率 (Hz)", row=i, col=2)
            fig.update_yaxes(title_text="幅度", row=i, col=1)
            fig.update_yaxes(title_text="幅度", row=i, col=2)

        fig.update_layout(
            title="FM调频调制解调系统",
            showlegend=False,
            height=1200,
            width=1600,
            font=dict(size=10)
        )

        return fig
    """
    def show_interactive(self, carrier_freq_range=None):
        """
        显示交互式图形
        """
        if not hasattr(self, 'interactive_fig') or self.interactive_fig is None:
            self.visualize(carrier_freq_range)
        self.interactive_fig.show()

if __name__ == '__main__':
    # 示例1: 简单正弦信号
    print("示例1: FM调制 - 简单正弦信号")
    message_signal1 = sin(2*pi*5*t)
    fm1 = FMModulation(message_signal1, carrier_freq=100,
                      sample_frequency=1000, T=2, freq_deviation=30)
    fm1.visualize(carrier_freq_range=(20, 100))
    fm1.show_interactive()
