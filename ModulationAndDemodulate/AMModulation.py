"""
AM调幅调制解调器
调制信号: s(t) = [A + m(t)] * cos(2πf_c*t)
其中 m(t) 是基带信号, A 是直流分量, f_c 是载波频率
"""
import numpy as np
import sympy as sp
from sympy import sin, cos, exp, pi, lambdify, latex
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from BasicFilter.BasicFilter import BasicFilter

t = sp.Symbol('t', real=True)


class AMModulation(BasicFilter):
    """
    AM调幅调制解调器
    调制信号: s(t) = [A + m(t)] * cos(2πf_c*t)
    其中 m(t) 是基带信号, A 是直流分量, f_c 是载波频率
    """
    def __init__(self, sym_exper,carrier_freq ,sample_frequency, T, modulation_index=0.5):
        """
        参数:
            sym_exper: sympy表达式，基带信号
            carrier_freq: 载波频率 (Hz)
            sample_frequency: 采样频率 (Hz)
            T: 信号时长 (秒)
            modulation_index: 调制指数 (0-1)
        """
        super().__init__(sym_exper, sample_frequency, T)
        self.modulation_index = modulation_index
        self.carrier_freq = carrier_freq
        # 生成载波信号
        self.carrier_signal = np.cos(2 * np.pi * carrier_freq * self.t_vals)

        # 归一化基带信号
        self.baseband_signal = self.func
        self.normalized_baseband = self.normalize_signal(self.baseband_signal)

        # 调制信号
        self.modulated_signal = None
        self.demodulated_signal = None

    def normalize_signal(self, signal):
        """归一化信号到[-1, 1]范围"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal

    def modulate(self):
        """
        AM调制
        s(t) = A_c * [1 + m * m(t)] * cos(2πf_c*t)
        """
        # 调制
        self.modulated_signal = (1 + self.modulation_index * self.normalized_baseband) * self.carrier_signal

        # 计算调制信号的FFT
        self.modulated_fft = np.fft.fft(self.modulated_signal)
        self.modulated_amplitude = np.abs(self.modulated_fft)

        return self.modulated_signal

    def demodulate(self, method='envelope'):
        """
        AM解调
        参数:
            method: 'envelope' (包络检波) 或 'coherent' (相干解调)
        """
        if self.modulated_signal is None:
            self.modulate()

        if method == 'envelope':
            # 包络检波：使用希尔伯特变换
            from scipy.signal import hilbert
            analytic_signal = hilbert(self.modulated_signal)
            envelope = np.abs(analytic_signal)

            # 去除直流分量
            self.demodulated_signal = envelope - np.mean(envelope)

        elif method == 'coherent':
            # 相干解调：与载波相乘后低通滤波
            # 与载波相乘
            demod_temp = self.modulated_signal * self.carrier_signal * 2

            # 低通滤波（频域截断）
            demod_fft = np.fft.fft(demod_temp)
            filter_response = np.zeros(len(self.fft_freq))
            # 低通滤波器截止频率设为载波频率的1/10
            cutoff = self.carrier_freq / 5
            filter_response[np.abs(self.fft_freq) <= cutoff] = 1.0

            filtered_fft = demod_fft * filter_response
            self.demodulated_signal = np.fft.ifft(filtered_fft).real

        # 计算解调信号的FFT
        self.demodulated_fft = np.fft.fft(self.demodulated_signal)
        self.demodulated_amplitude = np.abs(self.demodulated_fft)

        return self.demodulated_signal

    def visualize(self, demod_method='envelope'):
        """可视化调制解调过程"""
        # 执行调制和解调
        self.modulate()
        self.demodulate(method=demod_method)

        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'基带信号: {sp.pretty(self.sym_exper)}',
                '基带信号频谱',
                f'AM调制信号 (fc={self.carrier_freq}Hz, m={self.modulation_index})',
                '调制信号频谱',
                f'解调信号 ({demod_method})',
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

        # 更新布局
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

        fig.update_layout(
            title=f"AM调幅调制解调系统",
            showlegend=False,
            height=1200,
            width=1600,
            font=dict(size=10)
        )

        return fig


if __name__ == '__main__':
    # 示例1: 简单正弦信号
    print("示例1: AM调制 - 简单正弦信号")
    message_signal1 = sin(2*pi*5*t)
    am1 = AMModulation(message_signal1, carrier_freq=100,
                      sample_frequency=1000, T=2, modulation_index=0.8)
    fig1 = am1.visualize(demod_method='envelope')
    fig1.show()

    # 示例2: 复合信号
    print("\n示例2: AM调制 - 复合信号")
    message_signal2 = sin(2*pi*5*t) + 0.5*sin(2*pi*10*t)
    am2 = AMModulation(message_signal2, carrier_freq=100,
                      sample_frequency=1000, T=2, modulation_index=0.6)
    fig2 = am2.visualize(demod_method='coherent')
    fig2.show()

    # 示例3: 高调制指数
    print("\n示例3: AM调制 - 高调制指数")
    message_signal3 = sin(2*pi*3*t)
    am3 = AMModulation(message_signal3, carrier_freq=50,
                      sample_frequency=500, T=3, modulation_index=0.95)
    fig3 = am3.visualize(demod_method='envelope')
    fig3.show()
