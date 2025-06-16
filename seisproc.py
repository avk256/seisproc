import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.signal import butter, filtfilt, spectrogram
from scipy.stats.mstats import winsorize
from scipy.signal import hilbert, find_peaks
from scipy.signal import welch

from sklearn.preprocessing import StandardScaler


def plot_time_signals(df, fs):
    """
    Візуалізує часові сигнали зі всіх стовпців датафрейму.

    Parameters:
    - df: pandas.DataFrame — сигнал у 12 колонках (або інша кількість)
    - fs: float — частота дискретизації (Гц)
    """
    n_samples = df.shape[0]
    time = np.arange(n_samples) / fs

    n_cols = 3
    n_rows = int(np.ceil(len(df.columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        axes[i].plot(time, df[col], label=col)
        axes[i].set_title(f"Сигнал: {col}")
        axes[i].set_ylabel("Амплітуда")
        axes[i].grid(True)
        axes[i].legend()

        # Додаємо підпис осі X лише для нижнього ряду
        if i // n_cols == n_rows - 1:
            axes[i].set_xlabel("Час [с]")

    # Приховати зайві осі, якщо колонок менше
    for j in range(len(df.columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # return time

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

# Повторимо STA/LTA обчислення для моделювання Rayleigh-сигналу
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
