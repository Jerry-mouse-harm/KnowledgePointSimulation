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
    def __init__(self, sym_exper,sample_frequency, T, modulation_index=0.5, carrier_freq = None):
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

        # 归一化基带信号
        self.baseband_signal = self.func
        self.normalized_baseband = self.normalize_signal(self.baseband_signal)

        # 调制信号
        self.modulated_signal = None
        self.demodulated_signal = None

        # 图像
        self.interactive_fig = None

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
        # 生成载波信号
        self.carrier_signal = np.cos(2 * np.pi * self.carrier_freq * self.t_vals)

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

    def visualize(self, carrier_freq_range = None, demod_method='envelope'):
        """可视化调制解调过程，使用希尔伯特变换来实现"""
        # 判断载波信号的频率范围
        if carrier_freq_range is None:
            print('carrier_freq_range <UNK>')
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
                'AM调制信号',
                '调制信号频谱',
                f'解调信号 ({demod_method})',
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
        self.demodulate(method=demod_method)

        # 添加初始traces到各个subplot
        fig.add_trace(
            go.Scatter(x=self.t_vals, y=self.normalized_baseband,
                      mode='lines', name='基带信号',
                      line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.fft_freq, y=self.amplitude/self.N,
                      mode='lines', name='基带频谱',
                      line=dict(color='blue', width=1)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.t_vals, y=self.modulated_signal,
                      mode='lines', name='调制信号',
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.fft_freq, y=self.modulated_amplitude/self.N,
                      mode='lines', name='调制频谱',
                      line=dict(color='red', width=1)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.t_vals, y=self.demodulated_signal,
                      mode='lines', name='解调信号',
                      line=dict(color='green', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.fft_freq, y=self.demodulated_amplitude/self.N,
                      mode='lines', name='解调频谱',
                      line=dict(color='green', width=1)),
            row=3, col=2
        )

        # 为每个载波频率创建frames
        for i, cur_freq in enumerate(cur_carrier_freq):
            self.carrier_freq = cur_freq
            print(f'生成帧 {i+1}/{num_steps}: carrier_freq = {cur_freq:.2f} Hz')
            self.modulate()
            self.demodulate(method=demod_method)

            # 创建frame数据 - 注意：索引0-5对应6个subplot
            frames_data = [
                # trace 0: 基带信号 (row=1, col=1)
                go.Scatter(x=self.t_vals, y=self.normalized_baseband,
                          mode='lines', line=dict(color='blue', width=1.5)),

                # trace 1: 基带频谱 (row=1, col=2)
                go.Scatter(x=self.fft_freq, y=self.amplitude/self.N,
                          mode='lines', line=dict(color='blue', width=1)),

                # trace 2: 调制信号 (row=2, col=1)
                go.Scatter(x=self.t_vals, y=self.modulated_signal,
                          mode='lines', line=dict(color='red', width=1)),

                # trace 3: 调制频谱 (row=2, col=2)
                go.Scatter(x=self.fft_freq, y=self.modulated_amplitude/self.N,
                          mode='lines', line=dict(color='red', width=1)),

                # trace 4: 解调信号 (row=3, col=1)
                go.Scatter(x=self.t_vals, y=self.demodulated_signal,
                          mode='lines', line=dict(color='green', width=1.5)),

                # trace 5: 解调频谱 (row=3, col=2)
                go.Scatter(x=self.fft_freq, y=self.demodulated_amplitude/self.N,
                          mode='lines', line=dict(color='green', width=1))
            ]

            # 创建frame - 修复name命名
            frames.append(go.Frame(
                data=frames_data,
                name=str(i),  # ← 关键修复：使用简单的索引作为name
                layout=go.Layout(
                    title_text=f'AM调制解调系统 - 载波频率: {cur_freq:.2f} Hz'
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

    def show_interactive(self, carrier_freq_range=None):
        """
        显示交互式图形
        """
        if not hasattr(self, 'interactive_fig') or self.interactive_fig is None:
            self.visualize(carrier_freq_range)
        self.interactive_fig.show()


if __name__ == '__main__':
    # 示例1: 简单正弦信号
    print("示例1: AM调制 - 简单正弦信号")
    message_signal1 = sin(2*pi*5*t)
    am1 = AMModulation(message_signal1, sample_frequency=1000, T=2, modulation_index=0.8)
    am1.visualize(demod_method='envelope', carrier_freq_range=(20, 100))
    am1.show_interactive()
