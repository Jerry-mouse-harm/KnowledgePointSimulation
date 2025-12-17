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

    print("示例: AM调制")
    AM = AMModulation(exper, carrier_freq=100,
                       sample_frequency=1000, T=2, modulation_index=0.8)
    fig1 = AM.visualize(demod_method='envelope')
    fig1.show()

    print("示例: FM调制")
    FM = FMModulation(exper, carrier_freq=100,
                       sample_frequency=1000, T=2, freq_deviation=30)
    fig1 = FM.visualize()
    fig1.show()

    print("PM调制")
    PM = PMModulation(exper, carrier_freq=100,
                       sample_frequency=1000, T=2, phase_deviation=np.pi / 2)
    fig1 = PM.visualize()
    fig1.show()

