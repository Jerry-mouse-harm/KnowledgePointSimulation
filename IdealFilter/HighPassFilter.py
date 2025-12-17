import numpy as np
import sympy
import sympy as sp
from sympy import sin, cos, exp, pi
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from BasicFilter.BasicFilter import BasicFilter
t = sp.Symbol('t', real=True)
"""
实现高通滤波器
"""
class HighPassFilter(BasicFilter):
    def __init__(self, sym_exper , sample_frequency, T):
        super().__init__(sym_exper , sample_frequency, T)
        self.interactive_fig = None
        """设定截止频率"""
        self.cut_frequency = None
        """存储原有信号的数据"""
        self.old_func = self.func
        self.old_fft_vals = self.fft_vals
        self.old_fft_freq = self.fft_freq
        self.old_amplitude = self.amplitude
        self.old_phase = np.angle(self.fft_vals)

    def apply_lowpass_filter(self, cut_frequency):
        """
        应用理想高通滤波器（频域截断法）
        参数:
            cut_frequency: 截止频率 (Hz)

        返回:
            filtered_signal: 滤波后的时域信号
            filtered_fft: 滤波后的FFT结果
            filter_response: 滤波器频率响应
        """
        # 更新截止频率
        self.cut_frequency = cut_frequency

        # 创建理想高通通滤波器的频率响应（矩形窗）

        filter_response = np.zeros(len(self.fft_freq))
        # H(f) = 1 if |f| <= cutoff_freq, else 0
        # 这是np数组操作
        filter_response[np.abs(self.fft_freq) >= cut_frequency] = 1.0

        # 在频域应用滤波器（相乘）
        filtered_fft = self.old_fft_vals * filter_response

        # 通过逆FFT得到滤波后的时域信号
        filtered_signal = np.fft.ifft(filtered_fft).real

        return filtered_signal, filtered_fft, filter_response

    def create_interactive_plot(self, cutoff_range=None):
        """
        创建带有slider的交互式可视化界面

        参数:
            cutoff_range: tuple (min_cutoff, max_cutoff) 截止频率范围
                         如果为None，自动设置为 (0.1, sample_frequency/2)
        """
        if cutoff_range is None:
            cutoff_range = (0.1, self.sample_frequency / 2)

        min_cutoff, max_cutoff = cutoff_range

        # 生成多个截止频率值用于slider
        num_steps = 50
        cutoff_freqs = np.linspace(min_cutoff, max_cutoff, num_steps)

        # 创建子图布局：2x2 = 4个子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '滤波后时域信号',
                '滤波器频率响应',
                '滤波后频域幅度谱',
                '滤波后相位谱'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # 为每个截止频率生成数据
        frames = []
        for i, cutoff in enumerate(cutoff_freqs):
            filtered_signal, filtered_fft, filter_response = self.apply_lowpass_filter(cutoff)
            filtered_amplitude = np.abs(filtered_fft)
            filtered_phase = np.angle(filtered_fft)

            # 创建frame数据
            frame_data = [

                # (1,1) 滤波后时域信号
                go.Scatter(x=self.t_vals, y=filtered_signal,
                           mode='lines', name='滤波信号',
                           line=dict(color='red', width=1.5)),

                # (1,2) 滤波器频率响应
                go.Scatter(x=self.fft_freq, y=filter_response,
                           mode='lines', name='频率响应',
                           line=dict(color='green', width=2),
                           fill='tozeroy'),

                # (2,1) 滤波后频域幅度谱
                go.Scatter(x=self.fft_freq, y=filtered_amplitude / self.N,
                           mode='lines', name='滤波幅度谱',
                           line=dict(color='red', width=1)),

                # (2,2) 铝箔后的相位谱

                go.Scatter(x=self.fft_freq, y=filtered_phase,
                           mode='lines', name='滤波后相位',
                           line=dict(color='red', width=1.5),
                           showlegend=True),
            ]

            # 创建frame
            frames.append(go.Frame(
                data=frame_data,
                name=f'cutoff_{i}',
                layout=go.Layout(
                    title_text=f'截止频率: {cutoff:.2f} Hz'
                )
            ))

        # 添加初始数据（第一个frame的数据）
        for trace_idx, trace in enumerate(frames[0].data):
            # 确定trace应该在哪个subplot
            if trace_idx == 0:
                fig.add_trace(trace, row=1, col=1)
            elif trace_idx == 1:
                fig.add_trace(trace, row=1, col=2)
            elif trace_idx == 2:
                fig.add_trace(trace, row=2, col=1)
            elif trace_idx == 3:
                fig.add_trace(trace, row=2, col=2)

        fig.frames = frames

        # 创建slider
        sliders = [dict(
            active=0,
            yanchor="top",
            y=-0.05,
            xanchor="left",
            currentvalue=dict(
                prefix="截止频率: ",
                suffix=" Hz",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.05,
            steps=[dict(
                args=[[f'cutoff_{i}'],
                      dict(frame=dict(duration=0, redraw=True),
                           mode="immediate",
                           transition=dict(duration=0))],
                method="animate",
                label=f'{cutoff:.1f}'
            ) for i, cutoff in enumerate(cutoff_freqs)]
        )]

        # 更新布局
        fig.update_layout(
            title=dict(
                text=f'交互式高通滤波器 - 信号: x(t) = {sp.pretty(self.sym_exper)}',
                x=0.5,
                xanchor='center'
            ),
            sliders=sliders,
            height=900,
            width=1800,
            font=dict(size=10),
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="时间 (s)", row=1, col=1)
        fig.update_yaxes(title_text="幅度", row=1, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="|H(f)|", row=1, col=2)

        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="幅度", row=2, col=1)
        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=2)
        fig.update_yaxes(title_text="相位 (rad)", row=2, col=2)

        self.interactive_fig = fig

        return fig

    def show_interactive(self, cutoff_range=None):
        """
        显示交互式图形
        """
        if not hasattr(self, 'interactive_fig'):
            self.create_interactive_plot(cutoff_range)
        self.interactive_fig.show()

if __name__ == '__main__':
    # 示例1: 使用默认信号 sin(t)
    print("示例1: 高通滤波 sin(t)")
    filter1 = HighPassFilter(sin(t), 100, 4)
    filter1.create_interactive_plot((0.5, 20))
    filter1.show_interactive(cutoff_range=(0.5, 20))