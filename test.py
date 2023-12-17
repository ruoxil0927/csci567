import numpy as np
import matplotlib.pyplot as plt


def cal_curr(vt, vgs):
    def nonsat_curr(v):
        return kn * (2 * (vgs - vt) * v - v ** 2)
    kn = 0.148
    isat = kn * (vgs - vt) ** 2
    vds_sat = vgs - vt

    vds = np.linspace(0, 5, 100)
    id = np.piecewise(vds,
                      [vds <= vds_sat, vds > vds_sat],
                      [nonsat_curr, isat])
    return vds, id, vds_sat, isat


fig, ax = plt.subplots()
vds, id, vds_sat, isat = cal_curr(vt=0.8, vgs=3)
ax.plot(vds, id)
ax.scatter(vds_sat, isat)
vds, id, vds_sat, isat = cal_curr(vt=0.8, vgs=5)
ax.plot(vds, id)
ax.scatter(vds_sat, isat)
ax.set_xlabel('$V_{ds}$ (V)')
ax.set_ylabel('$I_D$ (mA)')

plt.show()



def cal_curr(vt=0.8):
    vgs = np.linspace(0, 5, 100)
    vds = 0.1
    kn = 0.148

    def sat_curr(vgs):
        return kn * (vgs - vt) ** 2

    def nonsat_curr(vgs):
        return kn * (2 * (vgs - vt) * vds - vds ** 2)

    id = np.piecewise(vgs, [vgs < vt, vt <= vgs, vgs >= 0.9], [0, sat_curr, nonsat_curr])

    return vgs, id

fig, ax = plt.subplots()
vgs, id = cal_curr(vt=0.8)
ax.plot(vgs, id)
ax.set_xlabel('$V_{gs}$ (V)')
ax.set_ylabel('$I_D$ (mA)')
plt.show()
