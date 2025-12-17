import sympy as sp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy import symbols, sin, cos, exp, pi, lambdify, latex

sp.init_printing()# 通过初始化printing，可以直观地展示函数
t = sp.Symbol('t', real=True)
class BasicFilter:
    def __init__(self,sym_exper , sample_frequency, T):
        # self.func = lambdify(t, sin(t), modules='numpy')
        func_symbolic = lambdify(t ,sym_exper ,modules='numpy')
        self.sym_exper = sym_exper
        self.sample_frequency = sample_frequency
        self.T = T
        self.N = int(self.T * self.sample_frequency)
        self.t_vals = np.linspace(0, self.T, self.N, endpoint=False)
        """self.func 是一个函数对象（通过 lambdify 创建），而不是数值数组。
        在第19行尝试对函数对象进行FFT时，np.fft.fft 期望接收一个数组，导致了 IndexError。"""
        self.func = func_symbolic(self.t_vals)
        # 计算傅里叶变换的频率和值
        self.fft_vals = np.fft.fft(self.func)
        self.fft_freq = np.fft.fftfreq(self.N, 1 / self.sample_frequency)
        # 计算相位谱和幅度谱
        self.amplitude = np.abs(self.fft_vals)
        self.phase = np.angle(self.fft_vals)
        # 实现中心化
        """
        self.mid_val = np.fft.fftshift(self.fft_vals)
        self.mid_freq = np.fft.fftshift(self.fft_freq)
        self.mid_amplitude = np.abs(self.mid_val / self.N)
        self.mid_phase = np.angle(self.mid_val)
        """
        # 用plotly创立画布
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '原始信号',
                '相位谱',
                '幅度谱',
                '中心化后的幅度谱'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )



    def show_primitive(self):
        fig = self.fig
        fig.add_trace(
            go.Scatter(
                x = self.t_vals,
                y = self.func,
                mode='lines',
                name='signal',
                line=dict(color='red', width=1)
            ),
            row=1,
            col=1
        )
        fig.update_xaxes(title_text="时间 (秒)", row=1, col=1)
        fig.update_yaxes(title_text="幅度 ", row=1, col=1)

    def show_phase(self):
        fig = self.fig
        fig.add_trace(
            go.Scatter(
                x = self.fft_freq,
                y = self.phase,
                mode='lines',
                name='phase',
                line=dict(color='yellow', width=1)
            ),
            row=1,
            col=2
        )
        fig.update_xaxes(title_text="频率 (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="相位 (弧度)", row=1, col=2)

    def show_amplitude(self):
        fig = self.fig
        fig.add_trace(
            go.Scatter(
                x = self.fft_freq,
                y = self.amplitude,
                mode='lines',
                name='amplitude',
                line=dict(color='blue', width=1)
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="幅度", row=2 , col=1)


    def show_mid_amplitude(self):
        mid_val = np.fft.fftshift(self.fft_vals)
        mid_freq = np.fft.fftshift(self.fft_freq)
        mid_amplitude = np.abs(mid_val / self.N)
        mid_phase = np.angle(mid_val)
        fig = self.fig
        fig.add_trace(
            go.Scatter(
                x = mid_freq,
                y = mid_amplitude,
                mode='lines',
                name='mid_amplitude',
                line=dict(color='purple', width=1)
            ),
            row=2,
            col=2
        )
        fig.update_xaxes(title_text="频率 (Hz)", row=2, col=2)
        fig.update_yaxes(title_text="幅度", row=2, col=2)




if __name__ == '__main__':
    test = BasicFilter(exp(5*t)+sin(t) , 200, 2)
    test.show_primitive()
    test.show_phase()
    test.show_amplitude()
    test.show_mid_amplitude()
    test.fig.update_layout(
        title=f"$ x(t) = {latex(test.sym_exper)} $",
        showlegend=False,
        height=800,
        width=1400,
        font=dict(size=11)
    )

    test.fig.show()
