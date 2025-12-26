from IdealFilter.BandStopFilter import BandStopFilter
from IdealFilter.BandPassFilter import BandPassFilter
from IdealFilter.HighPassFilter import HighPassFilter
from IdealFilter.LowPassFilter import LowPassFilter

from ModulationAndDemodulate.AMModulation import *
from ModulationAndDemodulate.FMModulation import *
from ModulationAndDemodulate.PMModulation import *

import sympy as sp
from sympy import *
t = sp.symbols('t', real=True)
if __name__ == '__main__':
    exper = sin(t)

    BP = BandPassFilter(exper, 100, 4, 5)
    BP.create_interactive_plot((0.5, 20))
    BP.show_interactive(cutoff_range=(0.5, 20))

    BS = BandStopFilter(exper, 100, 4, 5)
    BS.create_interactive_plot((0.5, 20))
    BS.show_interactive(cutoff_range=(0.5, 20))

    HP = HighPassFilter(exper, 100, 4)
    HP.create_interactive_plot((0.5, 20))
    HP.show_interactive(cutoff_range=(0.5, 20))

    LP = LowPassFilter(exper, 100, 4)
    LP.create_interactive_plot((0.5, 20))
    LP.show_interactive(cutoff_range=(0.5, 20))

    print("示例1: AM调制 - 简单正弦信号")
    message_signal1 = sin(2 * pi * 5 * t)
    am1 = AMModulation(message_signal1, sample_frequency=1000, T=2, modulation_index=0.8)
    am1.visualize(demod_method='envelope', carrier_freq_range=(20, 100))
    am1.show_interactive()

    print("示例1: FM调制 - 简单正弦信号")
    message_signal1 = sin(2 * pi * 5 * t)
    fm1 = FMModulation(message_signal1, carrier_freq=100,
                       sample_frequency=1000, T=2, freq_deviation=30)
    fm1.visualize(carrier_freq_range=(20, 100))
    fm1.show_interactive()

    print("PM调制")
    PM = PMModulation(exper, carrier_freq=100,
                       sample_frequency=1000, T=2, phase_deviation=np.pi / 2)
    fig1 = PM.visualize()
    fig1.show()

