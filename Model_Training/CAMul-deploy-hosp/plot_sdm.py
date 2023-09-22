import matplotlib.pyplot as plt
import numpy as np
import pudb
import scipy.fft as fft
import warnings
y = np.array([ -0.6903, -0.4755, -0.4666, -0.3771, -0.3502,
        -0.3234, -0.3323, -0.3771, -0.3950, -0.1712,  0.0256, -0.3144, -0.2249,
        -0.2160, -0.1086, -0.2965,  0.1330, -0.2160, -0.2518, -0.2697, -0.2160,
        -0.2339,  0.0435,  0.0077, -0.2518, -0.2428, -0.0370, -0.1981, -0.3681])

x = list(range(len(y)))

yp = np.array([-0.370, -0.1981, -0.3681])

vars =  np.array([-1.9723,-1.9014,-2.2895])

lb = yp - np.exp(vars)
ub = yp + np.exp(vars)
plt.plot(x, y, label="actual", color="mediumturquoise")
plt.plot(x[-3:],  yp, '--', label="predicted", color="darkorange")
plt.fill_between(x[-3:], lb, ub, color="bisque")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Case count")
plt.savefig("sdm1.png")
plt.clf()
y = np.array([-1.2356, -0.9330, -0.2671, -0.3276,
         0.2778,  0.0961, -0.9330, -0.6908, -0.3276,  0.0356,  0.0356, -0.2065,
         0.1567,  0.7015,  0.3383,  0.5199, -0.0855, -0.2065,  0.2778,  0.3988,
        -0.2671, -0.0249,  0.7015, -0.0249, -0.2671, -0.1460,  0.0356]
)

x = list(range(len(y)))

yp = np.array([-0.2256, 0.3502,-0.3149])

vars = np.array([-2.0982, -1.3916, -2.2454])

lb = yp - np.exp(vars)
ub = yp + np.exp(vars)
plt.plot(x, y, label="actual", color="mediumturquoise")
plt.plot(x[-3:],  yp, '--', label="predicted", color="darkorange")
plt.fill_between(x[-3:], lb, ub, color="bisque")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Case count")
plt.savefig("sdm2.png")

plt.clf()
X_ref = np.array([ 1533.40941176,  1355.99843137,  1329.33921569,  1248.99098039,
        1276.76117647,  1263.79647059,  1524.47019608,  2130.48588235,
        2729.98980392,  3272.08078431,  3740.54705882,  3925.3572549 ,
        3769.42313725,  3354.12156863,  3361.39333333,  2643.16588235,
        2541.61921569,  2499.92235294,  2165.82941176,  2497.80352941,
        2526.3027451 ,  2594.1227451 ,  2908.38235294,  3361.82980392,
        4160.10235294,  4880.41764706,  6573.03960784,  8658.77333333,
       10160.53764706, 10004.3827451 , 11584.26313725, 12767.22784314,
       13127.64980392, 11641.6345098 , 12767.93019608, 14671.98352941,
       13278.97215686, 10504.13960784,  9092.47803922,  7393.63372549,
        5756.73686275,  4184.81803922,  4227.78627451,  3699.96666667,
        3310.81411765,  3340.4145098 ,  3704.26313725,  3919.56705882,
        4112.63215686,  4142.22666667,  3555.65647059,  3099.41764706,
        2570.53686275,  2088.06313725,  1624.47215686,  1334.14941176,
         953.46666667,   914.96      ,   754.82117647,   798.79843137,
         872.27568627,  1203.87568627,  1940.82745098,  3118.3254902 ,
        4728.50705882,  6609.72901961,  7764.65176471,  8896.76588235,
        9242.90470588,  9758.36117647,  8865.26313725,  8671.24117647,
        7159.53215686,  6478.34784314,  5701.48196078,  5061.15176471,
        4418.18313725,  4326.97098039,  4390.41764706,  4825.2227451 ,
        5461.30352941,  4359.65921569,  7146.07568627,  7037.39607843,
        7642.93294118, 11188.32470588, 23165.37960784, 39823.58666667,
       45273.64941176, 40492.70705882, 30449.54627451, 17335.54156863,
       10562.91137255,  6206.63411765,  4002.39568627,  2922.02666667,
        2083.77098039,  1863.12313725,  1802.76980392,  1679.75686275,
        2033.78784314,  1946.03137255,  2749.71686275,  3028.77647059,
        4180.03529412,  4630.5745098 ,  6439.16392157,  5701.45686275,
        6191.28431373,  5906.47176471,  5891.84      ,  5718.64588235,
        6600.15960784,  5999.87333333,  7706.21960784,  7057.24509804,
        7649.6054902 ,  6419.69215686,  6605.15960784])



plt.plot(X_ref, label="Reference set in time domain", color="lightcoral")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Case count")
plt.savefig("sdm_GA.png")
plt.clf()
# def fftPlot(sig, dt=None, plot=True):
#     # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

#     if dt is None:
#         dt = 1
#         t = np.arange(0, sig.shape[-1])
#         xLabel = 'samples'
#     else:
#         t = np.arange(0, sig.shape[-1]) * dt
#         xLabel = 'freq [Hz]'

#     if sig.shape[0] % 2 != 0:
#         warnings.warn("signal preferred to be even in size, autoFixing it...")
#         t = t[0:-1]
#         sig = sig[0:-1]

#     sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

#     freq = np.fft.fftfreq(t.shape[0], d=dt)

#     # Plot analytic signal - right half of frequence axis needed only...
#     firstNegInd = np.argmax(freq < 0)
#     freqAxisPos = freq[0:firstNegInd]
#     sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

#     if plot:
#         plt.figure()
#         plt.plot(freqAxisPos, np.abs(sigFFTPos))
#         plt.xlabel(xLabel)
#         plt.ylabel('mag')
#         plt.title('Analytic FFT plot')
#         plt.savefig("sdm_GA_fft.png")

#     return sigFFTPos, freqAxisPos

# fftPlot(X_ref)
plt.plot(fft.fft(X_ref).real,label="Reference set in frequency domain",  color="slateblue")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.savefig("sdm_GA_fft.png")