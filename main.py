import sys
import time
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import solve_ivp


def r_critical(sigma: np.float64 = 10.0, b: np.float64 = 8.0 / 3.0) -> np.float64:
    r_crit = sigma * (sigma + b + 3.0) / (sigma - b - 1.0)
    if r_crit < 0.0:
        raise ValueError(
            "Error: Critical value of Rayleigh number expected to be positive!"
        )
    return r_crit


def systemLorenz(
    t: np.float64,
    state: np.ndarray,
    r: np.float64 = 28.0,
    sigma: np.float64 = 10.0,
    beta: np.float64 = 8.0 / 3.0,
) -> np.ndarray:
    x = state[0]
    y = state[1]
    z = state[2]
    ans = np.array(
        [sigma * (y - x), x * (r - z) - y, x * y - beta * z], dtype=np.float64
    )
    return ans


start_time = time.time()
x_start = 1.0
y_start = 0.0
z_start = 0.0
n = 10 ** 5
calc_time = 1000.0
dt = calc_time / n
r = 10.0
sigma = 2.0
beta = 0.5
r_crit = r_critical(sigma, beta)

print(f"Time step is {dt:.11e}")
print(f"Number of integration chunks is {n:.11e}")
print(f"Evaluation time: {calc_time}")
print(f"x0={x_start}, y0={y_start}, z0={z_start}")
print(f"r={r}, sigma={sigma}, b={beta}")
print(f"Critical Value of Rayleigh number is {r_crit}")

attractorLorenz = partial(systemLorenz, r=r, sigma=sigma, beta=beta)

state0 = np.array([x_start, y_start, z_start], dtype=np.float64)
t = np.linspace(0, calc_time, n)

states = solve_ivp(attractorLorenz, [0, calc_time], state0, t_eval=t)
end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.11e} s")

draw_phase = True
if draw_phase:
    fig1 = plt.figure()
    fig1.canvas.set_window_title("Lorenz Attractor")
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor\nr={r} σ={sigma} b={beta}")
    ax.plot(states.y[0, :], states.y[1, :], states.y[2, :], color="tomato", lw=0.5)

    fig2 = plt.figure(constrained_layout=True, figsize=(12.5, 4))
    fig2.canvas.set_window_title("X,Y,Z of time")
    spec2 = gridspec.GridSpec(
        ncols=3, nrows=1, figure=fig2, width_ratios=[3, 3, 3], height_ratios=[3]
    )
    f2_ax1 = fig2.add_subplot(spec2[0, 0])
    f2_ax1.set_title("x(t)")
    f2_ax1.set_xlabel("t")
    f2_ax1.set_ylabel("x(t)")
    f2_ax1.plot(t, states.y[0, :], color="red", lw=0.5)
    f2_ax2 = fig2.add_subplot(spec2[0, 1])
    f2_ax2.set_title("y(t)")
    f2_ax2.set_xlabel("t")
    f2_ax2.set_ylabel("y(t)")
    f2_ax2.plot(t, states.y[1, :], color="blue", lw=0.5)
    f2_ax3 = fig2.add_subplot(spec2[0, 2])
    f2_ax3.set_title("z(t)")
    f2_ax3.set_xlabel("t")
    f2_ax3.set_ylabel("y(t)")
    f2_ax3.plot(t, states.y[2, :], color="green", lw=0.5)

    fig3 = plt.figure(figsize=(14, 5))
    fig3.canvas.set_window_title("Projections")
    spec3 = gridspec.GridSpec(
        ncols=3, nrows=1, figure=fig2, width_ratios=[3, 3, 3], height_ratios=[3]
    )
    f3_ax1 = fig3.add_subplot(spec3[0, 0])
    f3_ax1.grid()
    f3_ax1.set_title("XY")
    f3_ax1.set_xlabel("x(t)")
    f3_ax1.set_ylabel("y(t)")
    f3_ax1.plot(states.y[0, :], states.y[1, :], color="red", lw=0.5)
    f3_ax2 = fig3.add_subplot(spec3[0, 1])
    f3_ax2.grid()
    f3_ax2.set_title("XZ")
    f3_ax2.set_xlabel("x(t)")
    f3_ax2.set_ylabel("z(t)")
    f3_ax2.plot(states.y[0, :], states.y[2, :], color="blue", lw=0.5)
    f3_ax3 = fig3.add_subplot(spec3[0, 2])
    f3_ax3.grid()
    f3_ax3.set_title("YZ")
    f3_ax3.set_xlabel("y(t)")
    f3_ax3.set_ylabel("z(t)")
    f3_ax3.plot(states.y[1, :], states.y[2, :], color="green", lw=0.5)
    plt.subplots_adjust(
        left=0.10, bottom=0.2, right=0.9, top=0.8, wspace=0.4, hspace=0.4
    )

    second = False
    if second:
        start_time = time.time()
        x_start = 1.0
        y_start = 0.0
        z_start = 0.0
        n = 10 ** 7
        calc_time = 100.0
        dt = calc_time / n
        r = 1500.0
        sigma = 2.0
        beta = 0.5
        r_crit = r_critical(sigma, beta)

        print(f"Time step is {dt:.11e}")
        print(f"Number of integration chunks is {n:.11e}")
        print(f"Evaluation time: {calc_time}")
        print(f"x0={x_start}, y0={y_start}, z0={z_start}")
        print(f"r={r}, sigma={sigma}, b={beta}")
        print(f"Critical Value of Rayleigh number is {r_crit}")
        state0 = np.array([x_start, y_start, z_start], dtype=np.float64)
        t = np.linspace(0, calc_time, n)

        states1 = solve_ivp(attractorLorenz, [0, calc_time], state0, t_eval=t)
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.11e} s")
        f2_ax1.plot(t, states1.y[0, :], color="black", linestyle="--", lw=1.0)
        f2_ax2.plot(t, states1.y[1, :], color="black", linestyle="--", lw=1.0)
        f2_ax3.plot(t, states1.y[2, :], color="black", linestyle="--", lw=1.0)
        f3_ax1.plot(
            states1.y[0, :], states1.y[1, :], color="black", lw=0.5, linestyle="--"
        )
        f3_ax2.plot(
            states1.y[0, :], states1.y[2, :], color="black", lw=0.5, linestyle="--"
        )
        f3_ax3.plot(
            states1.y[1, :], states1.y[2, :], color="black", lw=0.5, linestyle="--"
        )
    plt.show()

do_fourier = False
if do_fourier:
    skip = 10000
    x_fourier = fftshift(fft(states.y[0, skip:]))
    frequency = fftshift(fftfreq(n - skip, dt))
    fig3 = plt.figure(constrained_layout=True, figsize=(4, 8))
    fig3.canvas.set_window_title("Fourier Fast Transform")
    spec3 = gridspec.GridSpec(
        ncols=1, nrows=2, figure=fig3, width_ratios=[3], height_ratios=[3, 3]
    )
    f3_ax1 = fig3.add_subplot(spec3[0, 0])
    f3_ax1.set_title("x(t)")
    f3_ax1.set_xlabel("t")
    f3_ax1.set_ylabel("x(t)")
    f3_ax1.plot(t, states.y[0, :], color="red")

    f3_ax2 = fig3.add_subplot(spec3[1, 0])
    f3_ax2.set_title("Fourier Fast Transform")
    f3_ax2.set_xlabel("frequency")
    f3_ax2.set_ylabel("x_fourier")
    f3_ax2.axis(xmin=0.0, xmax=20.0)
    f3_ax2.plot(frequency[1:], 1.0 / (n - skip) * np.abs(x_fourier)[1:], color="blue")
    f3_ax2.grid()
    plt.show()
