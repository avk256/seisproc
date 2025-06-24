import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.signal import butter, filtfilt, spectrogram
from scipy.stats.mstats import winsorize
from scipy.signal import hilbert, find_peaks
from scipy.signal import welch
from scipy.signal import correlate
from scipy.signal import stft, istft, coherence, convolve2d

from sklearn.preprocessing import StandardScaler

import seaborn as sns


def plot_time_signals(df, fs, n_cols=4, threshold=0.5, verbose=False):
    """
    Візуалізує часові сигнали та виводить часи і значення всіх амплітуд, які перевищують поріг.

    Parameters:
    - df: pandas.DataFrame — сигнал у 12 колонках (або інша кількість)
    - fs: float — частота дискретизації (Гц)
    - threshold: float — порогове значення амплітуди (за модулем)
    """
    n_samples = df.shape[0]
    time = np.arange(n_samples) / fs

    #n_cols = 4
    n_rows = int(np.ceil(len(df.columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        signal = df[col].values
        above_thresh_idx = np.where(np.abs(signal) >= threshold)[0]
        peak_times = time[above_thresh_idx]
        peak_values = signal[above_thresh_idx]

        if (verbose):
            print(f"\nСигнал {col}: {len(peak_times)} значення(нь) вище порогу {threshold}")
            for t, v in zip(peak_times, peak_values):
                print(f"  Час = {t:.4f} с, Амплітуда = {v:.4f}")

        axes[i].plot(time, signal, label=col)
        axes[i].plot(peak_times, peak_values, 'ro', label='Піки > поріг')
        axes[i].set_title(f"Сигнал: {col}")
        axes[i].set_ylabel("Амплітуда")
        axes[i].set_xlabel("Час [с]")
        axes[i].grid(True)
        axes[i].legend()

    # Приховати зайві осі
    for j in range(len(df.columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Застосовує смуговий Butterworth-фільтр до сигналу.

    Parameters:
        data (np.ndarray): вхідний сигнал.
        lowcut (float): нижня межа смуги пропускання (Гц).
        highcut (float): верхня межа смуги пропускання (Гц).
        fs (float): частота дискретизації (Гц).
        order (int): порядок фільтра (типово 4).

    Returns:
        y (np.ndarray): відфільтрований сигнал.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def filter_dataframe(df, lowcut=20.0, highcut=200.0, fs=800, order=4):
    """
    Застосовує Butterworth bandpass-фільтр до кожного стовпця DataFrame з використанням apply.

    Parameters:
    - df: pandas.DataFrame з 12 сигналами
    - lowcut, highcut: межі смуги пропускання (Гц)
    - fs: частота дискретизації (Гц)
    - order: порядок фільтра

    Returns:
    - filtered_df: DataFrame з відфільтрованими сигналами
    """
    def apply_filter(column):
        return butter_bandpass_filter(column.values, lowcut, highcut, fs, order)

    filtered_df = df.apply(apply_filter, axis=0)
    filtered_df.index = df.index  # зберігаємо індекс оригінального DataFrame
    return filtered_df

def winsorize_dataframe(df, limits=(0.01, 0.01), inclusive=(True, True)):
    """
    Застосовує winsorization до кожного стовпця датафрейму.

    Parameters:
    - df: pandas.DataFrame — вхідний датафрейм з числовими стовпцями
    - limits: tuple — межі для нижньої та верхньої частини (наприклад, (0.01, 0.01) означає 1%)
    - inclusive: tuple — чи включати граничні значення (як у scipy winsorize)

    Returns:
    - winsorized_df: DataFrame з тими ж індексами та стовпцями
    """
    def apply_winsor(col):
        return winsorize(col, limits=limits, inclusive=inclusive).data

    winsorized_df = df.apply(apply_winsor, axis=0)
    winsorized_df.index = df.index  # зберігаємо індекси
    return winsorized_df

def geo_spectr_with_colorbar(data_ser, fs, n_samples, start_date, name=''):
    """
    Обчислює та візуалізує спектрограму сигналу із кольоровою шкалою.

    Parameters:
        data_ser (pd.Series): сигнал.
        fs (float): частота дискретизації (Гц).
        n_samples (int): кількість вибірок сигналу.
        start_date (str): початкова дата (використовується для індексації).
        name (str): назва сигналу (для заголовку графіка).

    Returns:
        signal (pd.Series): сигнал з часовим індексом.
    """
    # Формуємо часовий індекс
    time_index = pd.date_range(start_date, periods=n_samples, freq=f"{1000 // fs}ms")
    signal = pd.Series(data_ser.values, index=time_index)

    # Отримуємо сам сигнал
    x = signal.values

    nperseg = min(int(fs * 2), len(x))
    if len(x) == nperseg:
        nperseg = int(len(x)/6)
    # nperseg = int(fs*2)
    noverlap = int(nperseg * 0.9)
    # noverlap = min(nperseg * 0.5, int(nperseg * 0.9))

    print('nperseg', nperseg)
    print('noverlap', noverlap)

    # Спектрограма через scipy
    f, t, Sxx = spectrogram(x, fs=fs, window='hann', nperseg=nperseg,
                            noverlap=noverlap, mode='psd')

    # Перетворення в dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)  # додавання малих значень, щоб уникнути log(0)

    # Побудова
    plt.figure(figsize=(10, 6))
    im = plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
    cbar = plt.colorbar(im)
    cbar.set_label('Амплітуда [dB]')
    plt.ylabel('Частота [Hz]')
    plt.xlabel('Час [s]')
    plt.title('Спектр сигналу (FFT, dB) ' + name)
    plt.ylim(0, fs / 2)
    plt.tight_layout()
    plt.show()

    return signal

def spectr_plot(df, fs=800):
    """
    Будує спектрограми для всіх сигналів у датафреймі.

    Parameters:
        df (pandas.DataFrame): набір сигналів у стовпцях.
        fs (float): частота дискретизації (Гц).
    """
    # Побудова графіків кожної ознаки
    for column in df.columns:
        # _, _ = geo_spectr(df[column], fs=600, n_samples=len(df), start_date='2025-05-21', name=column)
        _ = geo_spectr_with_colorbar(df[column], fs=fs, n_samples=len(df), start_date='2025-05-21', name=column)

def compute_radial(V_X, V_Y1, V_Y2, theta_deg):
    """
    Обчислення радіальної компоненти сигналу.

    Parameters:
        V_X (np.ndarray): сигнал з осі X
        V_Y1 (np.ndarray): сигнал з осі Y1 (60° від X)
        V_Y2 (np.ndarray): сигнал з осі Y2 (-60° від X)
        theta_deg (float): кут до джерела (в градусах від осі X)

    Returns:
        V_R (np.ndarray): радіальна компонента сигналу
    """
    theta_rad = np.deg2rad(theta_deg)
    V_Y = (V_Y1 - V_Y2) / np.sqrt(3)
    V_R = V_X * np.cos(theta_rad) + V_Y * np.sin(theta_rad)
    return V_R

def psd_plot(signal, fs, name):
    """
    Будує графік спектральної щільності потужності (PSD) сигналу.

    Parameters:
        signal (np.ndarray): вхідний сигнал.
        fs (float): частота дискретизації (Гц).
        name (str): назва сигналу (для заголовку).
    """
    f, Pxx = welch(signal, fs=fs, nperseg=1024)

    plt.semilogy(f, Pxx)
    plt.title("Спектральна щільність потужності (PSD) " + name)
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Потужність / Гц")
    plt.grid(True)
    plt.show()

def extract_time_window(data, fs, time_range):
    """
    Виділяє підмножину сигналу за заданим часовим проміжком.

    Parameters:
        data (np.ndarray): одномірний масив даних сигналу
        fs (float): частота дискретизації в Гц
        time_range (tuple): кортеж (t_start, t_end) у секундах

    Returns:
        np.ndarray: зрізаний сигнал у заданому часовому проміжку
    """
    t_start, t_end = time_range
    i_start = int(t_start * fs)
    i_end = int(t_end * fs)
    return data[i_start:i_end]

def cut_time_window(Vr, Vz, fs, start_time, end_time):

    print(start_time)
    print(end_time)

    Vr_wind = extract_time_window(Vr, fs, (start_time, end_time))
    Vz_wind = extract_time_window(Vz, fs, (start_time, end_time))

    return Vr_wind, Vz_wind 


def sta_lta_trigger(signal, fs, sta_win=0.2, lta_win=1.0, on_thresh=3.0, off_thresh=1.5):
    """
    Визначає часові межі сигналу у шумовому середовищі на основі методу STA/LTA (Short-Term Average / Long-Term Average).
    Метод широко застосовується в сейсмології для детекції подій.

    Parameters:
        signal (np.ndarray): 1D масив сигналу.
        fs (float): частота дискретизації в Гц.
        sta_win (float): вікно короткострокового середнього (STA) у секундах.
        lta_win (float): вікно довгострокового середнього (LTA) у секундах.
        on_thresh (float): поріг активації (STA/LTA > порогу => старт події).
        off_thresh (float): поріг деактивації (STA/LTA < порогу => завершення події).

    Returns:
        ratio (np.ndarray): масив значень STA/LTA по часу.
        triggers (list of tuples): список тригерів у форматі ('start' або 'end', індекс).
    """
    sta_samples = int(sta_win * fs)
    lta_samples = int(lta_win * fs)

    squared = signal ** 2
    sta = np.convolve(squared, np.ones(sta_samples), 'same') / sta_samples
    lta = np.convolve(squared, np.ones(lta_samples), 'same') / lta_samples
    lta[lta == 0] = 1e-10
    ratio = sta / lta

    triggers = []
    triggered = False
    for i, r in enumerate(ratio):
        if not triggered and r > on_thresh:
            triggers.append(('start', i))
            triggered = True
        elif triggered and r < off_thresh:
            triggers.append(('end', i))
            triggered = False
    return ratio, triggers

# Використаємо вертикальну складову (Z)

def plot_sta_lta(signal, fs, sta_win=0.2, lta_win=1.0, on_thresh=3.0, off_thresh=1.5):

    ratio, triggers = sta_lta_trigger(signal, fs, sta_win=sta_win, lta_win=lta_win, on_thresh=on_thresh, off_thresh=off_thresh)

    # Побудуємо графік STA/LTA і позначимо тригери
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    n_samples = signal.shape[0]
    time = np.arange(n_samples) / fs


    # STA/LTA
    ax[0].plot(time, ratio, label='STA/LTA')
    ax[0].axhline(3.0, color='gray', linestyle='--', label='on_thresh')
    ax[0].axhline(1.5, color='lightgray', linestyle='--', label='off_thresh')
    for label, idx in triggers:
        color = 'green' if label == 'start' else 'red'
        ax[1].axvline(time[idx], color=color, linestyle='--')
        print(time[idx])
    ax[0].set_ylabel("STA / LTA")
    ax[0].set_xlabel("Час [с]")
    ax[0].legend()

    plt.tight_layout()
    plt.show()

    return triggers

def plot_hankel(Vr, Vz):
    """
    Будує 2D графік Ганкеля (еліптична траєкторія частинки) за радіальною та вертикальною компонентами.

    Parameters:
        Vr (array-like): радіальна компонента.
        Vz (array-like): вертикальна компонента.
    """
    plt.figure(figsize=(6, 6))
    # plt.scatter(Vr, Vz, s=1, alpha=0.7, label="Rayleigh ellipse (scatter)")
    plt.plot(Vr, Vz, label="Rayleigh ellipse (scatter)")
    plt.xlabel("Horizontal component Vx")
    plt.ylabel("Vertical component Vz")
    plt.title("Hankel Plot (Elliptical Particle Motion) — Scatter")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hankel_3d(Vr, Vz, fs, title='3D Hankel Plot'):
    """
    Побудова 3D графіку Ганкеля (еліптична траєкторія частинки)

    Parameters:
    -----------
    Vr : array-like
        Радіальна компонента швидкості (по осі X)
    Vz : array-like
        Вертикальна компонента швидкості (по осі Z)
    fs : float
        Частота дискретизації (Гц)
    title : str
        Назва графіка
    """
    Vr = np.asarray(Vr)
    Vz = np.asarray(Vz)

    if len(Vr) != len(Vz):
        raise ValueError("Vr та Vz повинні мати однакову довжину")

    # Побудова вісі часу
    t = np.arange(len(Vr)) / fs

    fig = go.Figure(data=go.Scatter3d(
        x=Vr,
        y=t,
        z=Vz,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Trajectory'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Vr (Radial Velocity)',
            yaxis_title='Time [s]',
            zaxis_title='Vz (Vertical Velocity)',
            aspectmode='cube'
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )

    fig.show()

def vpf(Vr, Vz, fs):
    """
    Виконує векторну поляризаційну фільтрацію (Vector Polarization Filtering) на основі Гілберт-перетворення.

    Parameters:
        Vr (np.ndarray): радіальна компонента сигналу.
        Vz (np.ndarray): вертикальна компонента сигналу.
        fs (float): частота дискретизації (Гц).

    Returns:
        imag_VPP (np.ndarray): уявна частина комплексної потужності (ознака еліптичної поляризації).
    """
    n_samples = Vr.shape[0]
    time = np.arange(n_samples) / fs

    # Обчислення аналітичних сигналів через Гілберт-перетворення
    Hr = hilbert(Vr)
    Hz = hilbert(Vz)

    # Комплексна потужність (vector polarization power)
    VPP = Hr.conj() * Hz
    imag_VPP = np.imag(VPP)

    # Візуалізація
    plt.figure(figsize=(12, 6))
    plt.plot(time, imag_VPP, label='Imaginary part of PHV (VPF output)')
    plt.title("Vector Polarization Filtering (VPF) Output")
    plt.xlabel("Time [s]")
    plt.ylabel("Imag(PHV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return imag_VPP


import numpy as np


def compute_multiple_snr_sta_lta(
    signal,
    fs,
    sta_window=0.05,
    lta_window=0.5,
    threshold=2.5,
    margin=0.1,
    min_gap_sec=0.5,
    signal_duration=0.2
):
    """
    Обчислює SNR для кількох подій у сейсмічному сигналі на основі STA/LTA.
    
    Parameters:
        signal (np.ndarray): 1D сигнал геофона.
        fs (float): частота дискретизації (Гц).
        sta_window (float): коротке вікно STA (сек).
        lta_window (float): довге вікно LTA (сек).
        threshold (float): поріг STA/LTA для виявлення сигналу.
        margin (float): додатковий буфер довкола події (сек).
        min_gap_sec (float): мінімальна відстань між подіями (сек).
        signal_duration (float): фіксована тривалість події (сек).
    
    Returns:
        List of tuples: (SNR_dB, (t_signal_start, t_signal_end), (t_noise_start, t_noise_end))
    """
    sta_n = int(sta_window * fs)
    lta_n = int(lta_window * fs)
    n = len(signal)

    # Енвелопа сигналу
    envelope = np.abs(hilbert(signal))

    # STA / LTA розрахунок
    sta = np.convolve(envelope, np.ones(sta_n) / sta_n, mode='same')
    lta = np.convolve(envelope, np.ones(lta_n) / lta_n, mode='same') + 1e-10
    ratio = sta / lta

    # Знаходимо піки STA/LTA
    min_gap_samples = int(min_gap_sec * fs)
    peaks, props = find_peaks(ratio, height=threshold, distance=min_gap_samples)

    results = []

    for peak in peaks:
        start_idx = max(0, peak - int(margin * fs))
        end_idx = min(n, peak + int((signal_duration + margin) * fs))

        # Вікно сигналу
        signal_seg = signal[start_idx:end_idx]
        signal_power = np.mean(signal_seg ** 2)

        # Вікно шуму перед подією
        noise_end = max(1, start_idx - 1)
        noise_start = max(0, noise_end - (end_idx - start_idx))
        noise_seg = signal[noise_start:noise_end]
        noise_power = np.mean(noise_seg ** 2) + 1e-12

        snr_db = 10 * np.log10(signal_power / noise_power)

        # snr_db = np.sqrt(signal_power) / np.sqrt(noise_power)

        results.append((
            snr_db,
            (start_idx / fs, end_idx / fs),
            (noise_start / fs, noise_end / fs)
        ))

    return results

def cross_corr(sig1, sig2, fs):
    """
    Обчислює кроскореляцію між двома сигналами та визначає часову затримку.

    Parameters:
        sig1 (np.ndarray): перший сигнал.
        sig2 (np.ndarray): другий сигнал.
        fs (float): частота дискретизації (Гц).

    Returns:
        lag (int): зсув у кількості відліків, що відповідає максимальній кореляції.
        time_delay (float): затримка в секундах (lag / fs).
        corr (np.ndarray): масив значень кроскореляції.
        lags (np.ndarray): масив відповідних зсувів у відліках.
    """
    corr = correlate(sig2, sig1, mode='full')
    lags = np.arange(-len(sig1) + 1, len(sig1))
    lag = lags[np.argmax(corr)]
    time_delay = lag / fs
    return lag, time_delay, corr, lags

def plot_cross_cor(sig1, sig2, fs, name1='', name2=''):
    """
    Візуалізує кроскореляцію між двома сигналами та виводить значення затримки.

    Parameters:
        sig1 (np.ndarray): перший сигнал.
        sig2 (np.ndarray): другий сигнал.
        fs (float): частота дискретизації (Гц).
        name1 (str): назва першого сигналу для підпису.
        name2 (str): назва другого сигналу для підпису.

    Returns:
        dt12 (float): затримка між сигналами у секундах.
    """
    lag12, dt12, corr12, lags12 = cross_corr(sig1, sig2, fs)

    print(f"Затримка між {name1} та {name2}: {dt12:.4f} с")

    # Візуалізація
    plt.figure(figsize=(10, 4))
    plt.plot(lags12 / fs, corr12, label=name1+" vs "+name2 )

    plt.axvline(dt12, color='r', linestyle='--', label="Δt12")

    plt.xlabel("Зсув (с)")
    plt.ylabel("Кроскореляція")
    plt.legend()
    plt.title("Кроскореляція сигналів")
    plt.grid()
    plt.show()

    # return lag12, dt12, corr12, lags12
    return dt12

def cross_corr_crossval_from_df(df, fs):
    """
    Виконує попарну кроскореляцію між сигналами різних геофонів для X, Y, Z компонент.

    Parameters:
        df (pd.DataFrame): датафрейм із колонками типу 'X1', 'Y11', 'Y12', 'Z1', ..., 'Z3'.
        fs (float): частота дискретизації.

    Returns:
        delays_X, delays_Y, delays_Z: словники із затримками між парами компонент.
    """

    # --- Вибір колонок по осях
    x_keys = [col for col in df.columns if col.startswith("X")]
    y_keys = [col for col in df.columns if col.startswith("Y")]
    z_keys = [col for col in df.columns if col.startswith("Z")]

    delays_X, delays_Y, delays_Z = {}, {}, {}

    # --- Попарні порівняння X-компонент
    for i in range(len(x_keys)):
        for j in range(len(x_keys)):
            k1, k2 = x_keys[i], x_keys[j]
            label1, label2 = k1, k2
            print(label1)
            print(label2)
            delays_X[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2)

    # --- Попарні порівняння Y-компонент
    for i in range(len(y_keys)):
        for j in range(len(y_keys)):
            k1, k2 = y_keys[i], y_keys[j]
            label1, label2 = k1, k2
            delays_Y[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2)

    # --- Попарні порівняння Z-компонент
    for i in range(len(z_keys)):
        for j in range(len(z_keys)):
            k1, k2 = z_keys[i], z_keys[j]
            label1, label2 = k1, k2
            delays_Z[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2)

    return delays_X, delays_Y, delays_Z


#def cros_corr_all(x_keys, y_keys, z_keys, fs):
    #"""
    #Обчислює та візуалізує попарну кроскореляцію для всіх сигналів осей X, Y, Z.

    #Parameters:
        #x_keys (list of str): ключі сигналів по осі X.
        #y_keys (list of str): ключі сигналів по осі Y.
        #z_keys (list of str): ключі сигналів по осі Z.
        #fs (float): частота дискретизації (Гц).

    #Returns:
        #delays_X, delays_Y, delays_Z (dict): словники із затримками між усіма парами сигналів для кожної осі.
            #Ключ: (ключ_1, ключ_2), Значення: затримка в секундах.
    #"""
    #--- Для X
    #x_keys = x_keys
    #delays_X = {}
    #for k1 in x_keys:
        #for k2 in x_keys:
            #label1 = f"eks1_X{k1[-6]}"
            #label2 = f"eks1_X{k2[-6]}"
            #delays_X[(k1, k2)] = plot_cross_cor(normalized_data[k1], normalized_data[k2], fs, label1, label2)

    #--- Для Z
    #z_keys = z_keys
    #delays_Z = {}
    #for k1 in z_keys:
        #for k2 in z_keys:
            #label1 = f"eks1_Z{k1[-6]}"
            #label2 = f"eks1_Z{k2[-6]}"
            #delays_Z[(k1, k2)] = plot_cross_cor(normalized_data[k1], normalized_data[k2], fs, label1, label2)

    #--- Для Y
    #y_keys = y_keys

    #delays_Y = {}
    #for k1 in y_keys:
        #for k2 in y_keys:
            #Витягаємо номер для формування мітки (наприклад, 'y1_21_norm' → '21')
            #label1 = f"eks1_Y{k1[3:5]}_norm"
            #label2 = f"eks1_Y{k2[3:5]}_norm"
            #delays_Y[(k1, k2)] = plot_cross_cor(normalized_data[k1], normalized_data[k2], fs, label1, label2)


    #return delays_X, delays_Y, delays_Z

def plot_delay_matrix(delays, title):
    """
    Візуалізує затримки між парами сигналів у вигляді теплової карти (матриці затримок).

    Parameters:
        delays (dict): словник із затримками між парами сигналів.
                       Формат ключів: (назва_сигналу_A, назва_сигналу_B), значення: затримка (float).
        title (str): заголовок графіка.

    Примітка:
        Функція підтримує асиметричний словник delays. Якщо (A, B) є, але (B, A) — ні,
        буде виведено лише відому затримку. Відсутні значення заповнюються NaN.
    """
    labels = sorted(set(i for pair in delays for i in pair))
    matrix = np.zeros((len(labels), len(labels)))

    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            key = (row, col)
            rev_key = (col, row)
            if key in delays:
                matrix[i, j] = delays[key]
            else:
                matrix[i, j] = np.nan  # якщо даних немає

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="coolwarm", center=0)
    plt.title(title)
    # plt.xlabel("Сигнал B")
    # plt.ylabel("Сигнал A")
    plt.show()
    
    return matrix
    

def plot_coherence(sig1, sig2, fs, name1, name2):
    """
    Обчислює та візуалізує когерентність (coherence) між двома сигналами у частотній області.

    Когерентність — це нормалізований показник схожості двох сигналів при різних частотах.
    Значення лежать у межах [0, 1], де 1 означає повну залежність на певній частоті.

    Parameters:
        sig1 (np.ndarray): перший сигнал.
        sig2 (np.ndarray): другий сигнал.
        fs (float): частота дискретизації (Гц).
        name1 (str): назва першого сигналу (для підпису графіка).
        name2 (str): назва другого сигналу.

    Returns:
        None. Виводить графік когерентності.
    """
    # Обчислення когерентності через метод Уелча
    f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=1024)

    # Побудова графіка в логарифмічному масштабі по осі Y
    plt.semilogy(f, Cxy)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Когерентність')
    plt.title(f'Когерентність між {name1} та {name2}')
    plt.grid(True)
    plt.show()
    
def coherent_subtraction_aligned_with_mask(sig_primary, 
                                           sig_reference, fs=600,
                                           nperseg=1024, noverlap=512,
                                           coherence_threshold=0.7):
    """
    Когерентне віднімання з урахуванням маски когерентності.
    Враховує затримку, фазу, амплітуду і обмежує віднімання тільки до когерентних частот.
    """

    # --- Внутрішні допоміжні функції ---
    def estimate_delay(sig1, sig2):
        """Оцінка затримки між сигналами через крос-кореляцію."""
        corr = np.correlate(sig1, sig2, mode='full')
        lag = np.argmax(corr) - len(sig2) + 1
        return lag

    def shift_signal(sig, delay):
        """Зсув сигналу на задану кількість семплів."""
        if delay > 0:
            return np.pad(sig, (delay, 0), mode='constant')[:len(sig)]
        elif delay < 0:
            return np.pad(sig, (0, -delay), mode='constant')[-delay:]
        else:
            return sig

    def normalize_signal(sig):
        """Нормалізація сигналу за амплітудою."""
        return sig / (np.max(np.abs(sig)) + 1e-10)

    # --- 1. Оцінка затримки ---
    delay = estimate_delay(sig_primary, sig_reference)

    # --- 2. Вирівнювання сигналів ---
    sig_ref_aligned = shift_signal(sig_reference, delay)

    # --- 3. STFT ---
    f, t_stft, Z_primary = stft(sig_primary, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Z_reference = stft(sig_ref_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # --- 4. Оцінка когерентності ---
    f_coh, Cxy = coherence(sig_primary, sig_ref_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Побудова маски когерентності
    coh_mask = (Cxy > coherence_threshold).astype(float)  # 1 – когерентна, 0 – ні
    coh_mask_2d = coh_mask[:, np.newaxis]  # для STFT (freq x time)

    # --- 5. Когерентне віднімання з маскою ---
    gain = np.abs(Z_primary) / (np.abs(Z_reference) + 1e-10)
    phase_correction = Z_reference / (np.abs(Z_reference) + 1e-10)
    noise_estimate = gain * phase_correction

    Z_clean = Z_primary - coh_mask_2d * noise_estimate  # застосування маски

    # --- 6. Обернене перетворення ---
    _, signal_cleaned = istft(Z_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # --- 7. Постобробка ---
    signal_cleaned = signal_cleaned[:len(sig_primary)]
    signal_cleaned = normalize_signal(signal_cleaned)

    return signal_cleaned, delay, coh_mask, f_coh
    


