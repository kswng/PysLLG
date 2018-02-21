# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt

from LLG_runge_kutta import numeric

PI = np.pi
sys_size = 10  # linear dimension of the square lattice
step = 0.01  # Numerical time step
current_time = 0.
alpha = 0.1  # Gilbert damping
J = -1  # exchange coupling, +1 for AFM, -1 for FM
Bext = [0, 0, 0.1]  # External magnetic field
calc_total = 500  #total calculation period in the unit of \hbar/|J|
calc_period = int(calc_total / step)
Temperature = 0.1 # temperature in the unit of |J|
fintemp = 1 # 1 for finite temperature calculation. 0 for zero temperature calculation
randseed = 0

np.random.seed(randseed)
system_sz = np.random.rand(sys_size + 2, sys_size + 2) - 0.5  # define sz
system_inplane = (1 - system_sz**2) * np.exp(2 * 1j * PI * np.random.rand(sys_size + 2, sys_size + 2)) # define sx and sy.


sxhist = []
syhist = []
szhist = []
timehist = []


for n in range(calc_period):

    system_sz, system_inplane = numeric(
        sys_size, system_sz, system_inplane, alpha, Bext, J, step, current_time, fintemp, Temperature)

    timehist.append(step * n)
    sxhist.append(np.mean(np.real(system_inplane)))
    syhist.append(np.mean(np.imag(system_inplane)))
    szhist.append(np.mean(system_sz))

    current_time = current_time + step


np.savetxt("data_out/sz.dat",system_sz)
np.savetxt("data_out/sx.dat",np.real(system_inplane))
np.savetxt("data_out/sy.dat",np.imag(system_inplane))

np.savetxt("data_out/sxhist.dat",sxhist)
np.savetxt("data_out/syhist.dat",syhist)
np.savetxt("data_out/szhist.dat",szhist)


plt.plot(timehist,sxhist)
plt.plot(timehist,syhist)
plt.plot(timehist,szhist)
plt.legend([r"$S_x$",r"$S_y$",r"$S_z$"])
plt.xlabel("time")
plt.show()