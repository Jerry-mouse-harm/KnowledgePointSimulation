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

    def visualize(self):
        """可视化调制解调过程"""
        # 执行调制和解调
        self.modulate()
        self.demodulate()

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

        # 1. 基带信号
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.normalized_baseband,
                                mode='lines', name='基带信号',
                                line=dict(color='blue', width=1.5)), row=1, col=1)

        # 2. 基带信号频谱
        fig.add_trace(go.Scatter(x=self.fft_freq, y=self.amplitude/self.N,
                                mode='lines', name='基带频谱',
                                line=dict(color='blue', width=1)), row=1, col=2)

        # 3. 调制信号
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.modulated_signal,
                                mode='lines', name='调制信号',
                                line=dict(color='red', width=1)), row=2, col=1)

        # 4. 调制信号频谱
        fig.add_trace(go.Scatter(x=self.fft_freq, y=self.modulated_amplitude/self.N,
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


if __name__ == '__main__':
    # 示例1: 简单正弦信号
    print("示例1: FM调制 - 简单正弦信号")
    message_signal1 = sin(2*pi*5*t)
    fm1 = FMModulation(message_signal1, carrier_freq=100,
                      sample_frequency=1000, T=2, freq_deviation=30)
    fig1 = fm1.visualize()
    fig1.show()

    # 示例2: 复合信号
    print("\n示例2: FM调制 - 复合信号")
    message_signal2 = sin(2*pi*5*t) + 0.5*sin(2*pi*10*t)
    fm2 = FMModulation(message_signal2, carrier_freq=100,
                      sample_frequency=1000, T=2, freq_deviation=40)
    fig2 = fm2.visualize()
    fig2.show()

    # 示例3: 低频信号
    print("\n示例3: FM调制 - 低频信号")
    message_signal3 = sin(2*pi*2*t) + 0.3*cos(2*pi*3*t)
    fm3 = FMModulation(message_signal3, carrier_freq=80,
                      sample_frequency=800, T=3, freq_deviation=25)
    fig3 = fm3.visualize()
    fig3.show()
