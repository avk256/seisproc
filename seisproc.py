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
from scipy.signal import detrend

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import plotly.subplots as psp
import plotly.graph_objs as go
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from collections import defaultdict
import plotly.express as px

def cut_dataframe_time_window(df, fs, start_time, end_time):
    """
    Вирізає сигнал у заданому часовому діапазоні з усіх колонок датафрейму.

    Parameters:
        df (pd.DataFrame): датафрейм із часовими рядами у кожній колонці
        fs (float): частота дискретизації в Гц
        start_time (float): початок інтервалу (в секундах)
        end_time (float): кінець інтервалу (в секундах)

    Returns:
        pd.DataFrame: новий датафрейм із вирізаними сигналами
    """
    i_start = int(start_time * fs)
    i_end = int(end_time * fs)
    return df.iloc[i_start:i_end].reset_index(drop=True)


def detrend_dataframe(df, type='linear'):
    """
    Застосовує scipy.signal.detrend до кожної колонки датафрейму.

    Parameters:
    - df: pandas.DataFrame — вхідні сигнали
    - type: str — тип детрендування ('linear' або 'constant')

    Returns:
    - pandas.DataFrame — детрендований датафрейм
    """
    df_detrended = pd.DataFrame(index=df.index)

    for col in df.columns:
        signal = df[col].values
        detrended_signal = detrend(signal, type=type)
        df_detrended[col] = detrended_signal

    return df_detrended

def plot_time_signals(df, fs, n_cols=4, columns=[], threshold=0.5, verbose=False, mode='plotly'):
    """
    Візуалізує часові сигнали та виводить часи і значення всіх амплітуд, які перевищують поріг.

    Parameters:
    - df: pandas.DataFrame — сигнал у стовпцях
    - fs: float — частота дискретизації (Гц)
    - n_cols: int — кількість колонок у сітці графіків
    - threshold: float — поріг амплітуди (по модулю)
    - verbose: bool — виводити часи та значення пік-амплітуд
    - mode: str — 'matplotlib' або 'plotly'
    """
    n_samples = df.shape[0]
    time = np.arange(n_samples) / fs
    n_signals = len(df.columns)
    n_rows = int(np.ceil(n_signals / n_cols))

    if mode == 'matplotlib':
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=True)
        axes = axes.flatten()

        for i, col in enumerate(columns):
            signal = df[col].values
            above_thresh_idx = np.where(np.abs(signal) >= threshold)[0]
            peak_times = time[above_thresh_idx]
            peak_values = signal[above_thresh_idx]

            if verbose:
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

        for j in range(n_signals, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    elif mode == 'plotly':
        fig = psp.make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True,
                                subplot_titles=[f"Сигнал: {col}" for col in columns], vertical_spacing=0.08)

        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            signal = df[col].values
            above_thresh_idx = np.where(np.abs(signal) >= threshold)[0]
            peak_times = time[above_thresh_idx]
            peak_values = signal[above_thresh_idx]

            if verbose:
                print(f"\nСигнал {col}: {len(peak_times)} значення(нь) вище порогу {threshold}")
                for t, v in zip(peak_times, peak_values):
                    print(f"  Час = {t:.4f} с, Амплітуда = {v:.4f}")

            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=col,
                                     showlegend=False), row=row, col=col_pos)
            fig.add_trace(go.Scatter(x=peak_times, y=peak_values, mode='markers',
                                     marker=dict(color='red', size=6), name='Піки > поріг',
                                     showlegend=False), row=row, col=col_pos)
            
            fig.update_xaxes(
            title_text="Час [с]",
            showticklabels=True,
            row=row, col=col_pos)

        if n_cols>1:
            fig.update_layout(
                height=400 * n_rows,
                width=400 * n_cols,
                title_text="Часові сигнали з позначенням піків",
                showlegend=False
            )
            
        
        
        
        if n_cols==1:
            fig.update_layout(
                height=1600 * n_rows,
                width=1200 * n_cols,
                title_text="Часові сигнали з позначенням піків",
                showlegend=False
            )

        fig.update_xaxes(title_text="Час [с]")
        fig.update_yaxes(title_text="Амплітуда")

        # Відобразити повний тулбар Plotly
        fig.show(config=dict(displayModeBar=True, responsive=True))

    elif mode == 'plotly_one':
        # Групування за першим символом назви
        groups = defaultdict(list)
        for col in columns:
            key = col[0]  # перший символ
            groups[key].append(col)

        fig = go.Figure()
        color_palette = px.colors.qualitative.Set1
        color_idx = 0

        # for key, group_cols in sorted(groups.items()):
        for col in columns:
            signal = df[col].values
            fig.add_trace(go.Scatter(
                x=time,
                y=signal,
                mode='lines',
                name=f"{col}",
                line=dict(width=1),
                # legendgroup=key,
                showlegend=True
            ))
        color_idx += 1

        fig.update_layout(
            title="Сигнали, згруповані за першою літерою назви вісі геофона",
            xaxis_title="Час [с]",
            yaxis_title="Амплітуда",
            height=500 + 40 * len(groups),
            width=1200,
            template="plotly_white"
        )
        fig.show(config=dict(displayModeBar=True, responsive=True))
        return fig

    else:
        raise ValueError("mode має бути 'matplotlib', 'plotly' або 'plotly_one'")
    
    return fig

    
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

# def geo_spectr_with_colorbar(data_ser, fs, n_samples, start_date, name='', ax=None):
#     """
#     Обчислює та візуалізує спектрограму сигналу із кольоровою шкалою.

#     Parameters:
#         data_ser (pd.Series): сигнал.
#         fs (float): частота дискретизації (Гц).
#         n_samples (int): кількість вибірок сигналу.
#         start_date (str): початкова дата (використовується для індексації).
#         name (str): назва сигналу (для заголовку графіка).
#         ax (matplotlib.axes.Axes or None): вісь для побудови. Якщо None — створюється нова фігура.

#     Returns:
#         signal (pd.Series): сигнал з часовим індексом.
#     """
#     # Формуємо часовий індекс
#     time_index = pd.date_range(start_date, periods=n_samples, freq=f"{1000 // fs}ms")
#     signal = pd.Series(data_ser.values, index=time_index)

#     x = signal.values
#     nperseg = min(int(fs * 2), len(x))
#     if len(x) == nperseg:
#         nperseg = int(len(x) / 6)
#     noverlap = int(nperseg * 0.9)

#     f, t, Sxx = spectrogram(x, fs=fs, window='hann', nperseg=nperseg,
#                             noverlap=noverlap, mode='psd')
#     Sxx_dB = 10 * np.log10(Sxx + 1e-12)

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 6))

#     im = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud')
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label('Амплітуда [dB]')
#     ax.set_ylabel('Частота [Hz]')
#     ax.set_xlabel('Час [s]')
#     ax.set_title('Спектр сигналу (FFT, dB) ' + name)
#     ax.set_ylim(0, fs / 2)

#     if ax is None:
#         plt.tight_layout()
#         plt.show()

#     return signal

def geo_spectr_with_colorbar(data_ser, fs, n_samples, start_date, name='',
                             ax=None, seg_len_s=None, overlap_s=None):
    """
    Обчислює та візуалізує спектрограму сигналу із кольоровою шкалою.

    Parameters:
        data_ser (pd.Series): сигнал.
        fs (float): частота дискретизації (Гц).
        n_samples (int): кількість вибірок сигналу.
        start_date (str): початкова дата (використовується для індексації).
        name (str): назва сигналу (для заголовку графіка).
        ax (matplotlib.axes.Axes or None): вісь для побудови. Якщо None — створюється нова фігура.
        seg_len_s (float or None): довжина сегмента в секундах для спектрограми (nperseg = seg_len_s * fs).
        overlap_s (float or None): перекриття в секундах (noverlap = overlap_s * fs).

    Returns:
        signal (pd.Series): сигнал з часовим індексом.
    """
    # Формуємо часовий індекс
    time_index = pd.date_range(start=start_date, periods=n_samples, freq=f"{1000 // fs}ms")
    signal = pd.Series(data_ser.values, index=time_index)

    x = signal.values

    # Обчислюємо nperseg та noverlap
    if seg_len_s is not None:
        nperseg = int(seg_len_s * fs)
    else:
        nperseg = min(int(fs * 2), len(x))
        if len(x) == nperseg:
            nperseg = int(len(x) / 6)

    if overlap_s is not None:
        noverlap = int(overlap_s * fs)
    else:
        noverlap = int(nperseg * 0.9)

    # Захист від помилки: noverlap must be less than nperseg
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    print('spectr')
    print(nperseg)
    print(noverlap)

    # Спектрограма
    f, t, Sxx = spectrogram(x, fs=fs, window='hann',
                            nperseg=nperseg, noverlap=noverlap, mode='psd')
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Амплітуда [dB]')
    ax.set_ylabel('Частота [Hz]')
    ax.set_xlabel('Час [s]')
    ax.set_title('Спектр сигналу (FFT, dB) ' + name)
    ax.set_ylim(0, fs / 2)

    if ax is None:
        plt.tight_layout()
        plt.show()

    return signal




def spectr_plot(df, fs=800, n_cols=4, columns=['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], seg_len_s=None, overlap_s=None):
    """
    Будує спектрограми для вказаних сигналів у DataFrame.

    Parameters:
        df (pandas.DataFrame): набір сигналів у стовпцях.
        fs (float): частота дискретизації (Гц).
        columns (list or None): список назв колонок для побудови. Якщо None — використовуються всі.
        n_cols (int): кількість графіків у рядку.
    """
    if columns is None:
        columns = df.columns.tolist()

    n = len(columns)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        # Припускаємо, що geo_spectr_with_colorbar приймає параметр ax
        geo_spectr_with_colorbar(df[col], fs=fs, n_samples=len(df), start_date='2025-05-21', name=col, ax=ax, seg_len_s=seg_len_s, overlap_s=seg_len_s)
        # geo_spectr_with_colorbar(df[col], fs=fs, n_samples=len(df), start_date='2025-05-21', name=col)

    # Сховати зайві осі
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
    return fig
    
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

def psd_plot_df(df, fs, n_cols=4, columns=['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], mode='matplotlib', scale='db'):
    """
    Візуалізує графіки спектральної щільності потужності (PSD) для вказаних колонок DataFrame
    з використанням matplotlib або plotly.

    Parameters:
        df (pd.DataFrame): вхідний датафрейм із сигналами.
        fs (float): частота дискретизації (Гц).
        n_cols (int): кількість колонок у макеті subplot.
        columns (list): список назв колонок (за замовчуванням усі, що починаються з X/Y/Z).
        backend (str): 'matplotlib' або 'plotly'.

    Returns:
        fig: matplotlib.figure.Figure або plotly.graph_objects.Figure
    """
    if columns is None:
        columns = [col for col in df.columns if col.startswith(('X', 'Y', 'Z'))]

    n = len(columns)
    n_rows = (n + n_cols - 1) // n_cols

    if mode == 'matplotlib':
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)
        axes = axes.flatten()

        for i, col in enumerate(columns):
            signal = df[col].values
            f, Pxx = welch(signal, fs=fs, nperseg=1024)
            Pxx_dB = 10 * np.log10(Pxx + 1e-12)  # захист від log(0)
            if scale=='db':
                axes[i].semilogy(f, Pxx_dB)
            if scale=='energy':
                axes[i].semilogy(f, Pxx)
            axes[i].set_title(f"PSD: {col}")
            axes[i].set_xlabel("Частота (Гц)")
            axes[i].set_ylabel("Потужність / Гц, Дб")
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.tight_layout()
        return fig

    elif mode == 'plotly':
        fig = psp.make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=[f"PSD: {col}" for col in columns],
                            shared_xaxes=True)

        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            signal = df[col].values
            f, Pxx = welch(signal, fs=fs, nperseg=1024)
            Pxx_dB = 10 * np.log10(Pxx + 1e-12)  # захист від log(0)
            if scale=='db':
                fig.add_trace(
                    go.Scatter(x=f, y=Pxx_dB, mode='lines', name=col),
                    row=row, col=col_pos
                )
            if scale=='energy':
                fig.add_trace(
                    go.Scatter(x=f, y=Pxx, mode='lines', name=col),
                    row=row, col=col_pos
                )
            fig.update_xaxes(
            title_text="Частота (Гц)",
            showticklabels=True,
            row=row, col=col_pos)

        fig.update_layout(
            height=300 * n_rows,
            width=400 * n_cols,
            title_text="Спектральна щільність потужності (PSD)",
            showlegend=False
        )
        
        if scale=='db':
            fig.update_yaxes(title_text="Потужність / Гц, Дб", tickformat=".2f")
        if scale=='energy':
            fig.update_yaxes(title_text="Потужність / Гц, Дб", tickformat=".2e")
        fig.update_xaxes(title_text="Частота (Гц)")
        return fig
    
    elif mode == 'matrix':
        f_list = []
        Pxx_list = []
        Pxx_dB_list = []
        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            signal = df[col].values
            f, Pxx = welch(signal, fs=fs, nperseg=1024)
            Pxx_dB = 10 * np.log10(Pxx + 1e-12)  # захист від log(0)
            f_list.append(f)
            Pxx_list.append(Pxx)
            Pxx_dB_list.append(Pxx_dB)
        if scale=='db':
            res = Pxx_dB_list
        if scale=='energy':
            res = Pxx_list
        return f_list, res 
        

    else:
        raise ValueError("backend повинен бути 'matplotlib' або 'plotly'")

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

def plot_hankel(Vr, Vz, scale=1.0, mode='matplotlib', start_time=None, end_time=None, fs=800):
    """
    Створює графік Ганкеля (еліптична траєкторія частинки) у matplotlib або plotly.

    Parameters:
        Vr (array-like): радіальна компонента сигналу.
        Vz (array-like): вертикальна компонента сигналу.
        scale (float): масштаб (1.0 = стандартний) — впливає на розмір графіка та шрифтів.
        mode (str): 'matplotlib' або 'plotly' — вибір бібліотеки візуалізації.

    Returns:
        fig: об'єкт Figure matplotlib або plotly.
    """
    Vr = np.asarray(Vr)
    Vz = np.asarray(Vz)
    
    if start_time is not None and end_time is not None:
       i_start = int(start_time * fs)
       i_end = int(end_time * fs)
       Vr = Vr[i_start:i_end]
       Vz = Vz[i_start:i_end]

    if mode == 'matplotlib':
        base_size = 6
        fig_size = (base_size * scale, base_size * scale)
        font_size = 10 * scale

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(Vr, Vz, label="Rayleigh ellipse", linewidth=1.5 * scale)
        ax.set_xlabel("Horizontal component Vr", fontsize=font_size)
        ax.set_ylabel("Vertical component Vz", fontsize=font_size)
        ax.set_title("Hankel Plot (Elliptical Particle Motion)", fontsize=font_size + 2)
        ax.grid(True)
        ax.axis('equal')
        ax.tick_params(labelsize=font_size * 0.9)
        ax.legend(fontsize=font_size)
        fig.tight_layout()
        return fig

    elif mode == 'plotly':
        fig_width = 500 * scale
        fig_height = 500 * scale

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Vr, y=Vz,
            mode='lines',
            name='Rayleigh ellipse',
            line=dict(width=2)
        ))

        fig.update_layout(
            width=fig_width,
            height=fig_height,
            title="Hankel Plot (Elliptical Particle Motion)",
            xaxis_title="Horizontal component Vr",
            yaxis_title="Vertical component Vz",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    else:
        raise ValueError("mode має бути 'matplotlib' або 'plotly'")

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

def vpf(Vr, Vz, fs, mode='matrix', scale=1.0):
    """
    Виконує векторну поляризаційну фільтрацію (VPF) та візуалізує уявну частину комплексної потужності.

    Parameters:
        Vr (np.ndarray): радіальна компонента сигналу.
        Vz (np.ndarray): вертикальна компонента сигналу.
        fs (float): частота дискретизації (Гц).
        mode (str): 'matrix' — повертає масив; 'fig' — повертає matplotlib.figure; 'plotly' — повертає plotly.graph_objects.Figure.
        scale (float): масштаб (тільки для matplotlib), за замовчуванням 1.0.

    Returns:
        або: np.ndarray (imag_VPP), або matplotlib.figure.Figure, або plotly.graph_objects.Figure
    """
    n_samples = Vr.shape[0]
    time = np.arange(n_samples) / fs

    # Обчислення аналітичних сигналів
    Hr = hilbert(Vr)
    Hz = hilbert(Vz)

    VPP = Hr * Hz.conj()
    imag_VPP = np.imag(VPP)

    if mode == 'matrix':
        return imag_VPP

    elif mode == 'fig':
        base_size = 6
        fig_size = (base_size * scale * 2, base_size * scale)
        font_size = 10 * scale

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(time, imag_VPP, label='Imaginary part of PHV (VPF output)', linewidth=1.2 * scale)
        ax.set_xlabel("Time [s]", fontsize=font_size)
        ax.set_ylabel("Imag(PHV)", fontsize=font_size)
        ax.grid(True)
        ax.legend(fontsize=font_size)
        ax.tick_params(labelsize=font_size * 0.9)
        ax.set_title("Vector Polarization Filtering (VPF) Output", fontsize=font_size + 2)
        fig.tight_layout()
        return fig

    elif mode == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=imag_VPP,
            mode='lines',
            name='Imaginary part of PHV',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title="Vector Polarization Filtering (VPF) Output",
            xaxis_title="Time [s]",
            yaxis_title="Imag(PHV)",
            height=400,
            width=900,
            margin=dict(t=40, b=40, l=60, r=40),
        )
        return fig

    else:
        raise ValueError("mode має бути 'matrix', 'fig', або 'plotly'")

def energy_density(Vr, Vz, rho):
    """
    Виконує векторну поляризаційну фільтрацію (VPF) та візуалізує уявну частину комплексної потужності.

    Parameters:
        Vr (np.ndarray): радіальна компонента сигналу.
        Vz (np.ndarray): вертикальна компонента сигналу.
        fs (float): частота дискретизації (Гц).
        mode (str): 'matrix' — повертає масив; 'fig' — повертає matplotlib.figure; 'plotly' — повертає plotly.graph_objects.Figure.
        scale (float): масштаб (тільки для matplotlib), за замовчуванням 1.0.

    Returns:
        або: np.ndarray (imag_VPP), або matplotlib.figure.Figure, або plotly.graph_objects.Figure
    """
   
    # Обчислення аналітичних сигналів
    Hr = hilbert(Vr)
    Hz = hilbert(Vz)

    VPP = Hr.conj() * Hz / 2
    imag_VPP = np.imag(VPP)
    print(imag_VPP)
    ls = 10 * np.log(rho*np.abs(imag_VPP))
    
    return np.mean(ls)

    
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
    
def cross_corr(sig1, sig2, fs, allowed_lag_ranges_s=None):
    """
    Обчислює кроскореляцію між двома сигналами та визначає затримку
    лише в межах заданих інтервалів лагів (у секундах).

    Parameters:
        sig1 (np.ndarray): перший сигнал.
        sig2 (np.ndarray): другий сигнал.
        fs (float): частота дискретизації (Гц).
        allowed_lag_ranges_s (list of tuple): список дозволених інтервалів затримок у секундах, наприклад:
            [(-0.05, -0.03), (-0.005, 0.005), (0.03, 0.05)]

    Returns:
        lag (int or None): зсув у кількості відліків (None, якщо не знайдено).
        time_delay (float or None): затримка в секундах (lag / fs).
        corr (np.ndarray): масив значень кроскореляції (тільки в дозволених інтервалах).
        lags (np.ndarray): масив відповідних зсувів у відліках.
    """

    # Повна кроскореляція між сигналами
    corr_full = correlate(sig2, sig1, mode='full')
    total_lags = np.arange(-len(sig1) + 1, len(sig2))

    # Якщо не задано allowed_lag_ranges_s — використовуємо всю шкалу
    if allowed_lag_ranges_s is None:
        mask = np.ones_like(total_lags, dtype=bool)
    else:
        # Створюємо маску, яка залишає тільки лаги в дозволених діапазонах
        mask = np.zeros_like(total_lags, dtype=bool)
        for lag_range in allowed_lag_ranges_s:
            min_lag = int(np.floor(lag_range[0] * fs))
            max_lag = int(np.ceil(lag_range[1] * fs))
            mask |= (total_lags >= min_lag) & (total_lags <= max_lag)

    # Застосовуємо маску
    lags = total_lags[mask]
    corr = corr_full[mask]

    # Якщо в дозволеному діапазоні немає лагів — повертаємо None
    if len(lags) == 0:
        return None, None, np.array([]), np.array([])

    # Знаходимо лаг із максимумом кореляції
    max_index = np.argmax(corr)
    lag = lags[max_index]
    time_delay = lag / fs

    return lag, time_delay, corr, lags

def plot_cross_cor(sig1, sig2, fs, name1='', name2='', verbose=False, allowed_lag_ranges_s=None):
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
    lag12, dt12, corr12, lags12 = cross_corr(sig1, sig2, fs, allowed_lag_ranges_s=allowed_lag_ranges_s)

    #print(f"Затримка між {name1} та {name2}: {dt12:.4f} с")

    # Візуалізація
    if verbose:
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

def cross_corr_crossval_from_df(df, fs, verbose=False, allowed_lag_ranges_s=None):
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
        for j in range(i+1, len(x_keys)):
            k1, k2 = x_keys[i], x_keys[j]
            label1, label2 = k1, k2
            #print(label1)
            #print(label2)
            delays_X[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2, verbose=verbose, allowed_lag_ranges_s=allowed_lag_ranges_s)

    # --- Попарні порівняння Y-компонент
    for i in range(len(y_keys)):
        for j in range(i+1,len(y_keys)):
            k1, k2 = y_keys[i], y_keys[j]
            label1, label2 = k1, k2
            delays_Y[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2, verbose=verbose, allowed_lag_ranges_s=allowed_lag_ranges_s)

    # --- Попарні порівняння Z-компонент
    for i in range(len(z_keys)):
        for j in range(i+1,len(z_keys)):
            k1, k2 = z_keys[i], z_keys[j]
            label1, label2 = k1, k2
            delays_Z[(k1, k2)] = plot_cross_cor(df[k1].values, df[k2].values, fs, label1, label2, verbose=verbose, allowed_lag_ranges_s=allowed_lag_ranges_s)

    return delays_X, delays_Y, delays_Z

def plot_delay_matrix(delays, title, mode='matrix'):
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
                
                
    fig, ax = plt.subplots()
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="coolwarm", center=0)

    ax.set_title(title)
  
    ret = matrix
    
    if mode == 'fig':
    
        ret = fig

    return ret    

def eval_delay_matrix(delays, labels=None):
    """
    Побудова матриці затримок (без plt.subplots), щоб працювати з subplot.
    """
    if labels is None:
        labels = sorted(set(i for pair in delays for i in pair))
        
    matrix = np.full((len(labels), len(labels)), np.nan)

    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            key = (row, col)
            if key in delays:
                matrix[i, j] = delays[key]

    return matrix, labels


def plot_multiple_delay_matrices(all_delays: dict, title_prefix="Матриця", cols=2, cmap="coolwarm", fmt=".2f"):
    """
    Візуалізує кілька матриць затримок на одній фігурі з використанням тільки matplotlib.
    """
    plt.close('all')  # уникнення дублювання

    n = len(all_delays)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    vmin = min(np.nanmin(eval_delay_matrix(d)[0]) for d in all_delays.values())
    vmax = max(np.nanmax(eval_delay_matrix(d)[0]) for d in all_delays.values())

    for i, (name, delay_dict) in enumerate(all_delays.items()):
        matrix, labels = eval_delay_matrix(delay_dict)
        ax = axes[i]

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(f"{title_prefix}: {name}")

        # Підписи значень в клітинках
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                val = matrix[y, x]
                if not np.isnan(val):
                    color = "white" if abs(val) > (vmin + vmax) / 2 else "black"
                    ax.text(x, y, format(val, fmt), ha="center", va="center", color=color, fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Видалити зайві subplot-и
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Затримка (с)")
    fig.tight_layout()
    return fig

def plot_coherence(sig1, sig2, fs, name1, name2, mode='matplotlib', scale=1.0):
    """
    Обчислює та візуалізує когерентність між двома сигналами у частотній області.

    Parameters:
        sig1 (np.ndarray): перший сигнал.
        sig2 (np.ndarray): другий сигнал.
        fs (float): частота дискретизації (Гц).
        name1 (str): назва першого сигналу (для підпису графіка).
        name2 (str): назва другого сигналу.
        mode (str): 'matplotlib' або 'plotly' — режим побудови.
        scale (float): масштаб фігури (тільки для matplotlib).

    Returns:
        fig: matplotlib.figure.Figure або plotly.graph_objects.Figure
    """
    # Обчислення когерентності
    f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=1024)

    if mode == 'matplotlib':
        base_size = 6
        fig_size = (base_size * scale * 1.5, base_size * scale)
        font_size = 10 * scale

        fig, ax = plt.subplots(figsize=fig_size)
        ax.semilogy(f, Cxy, color='blue', linewidth=1.5 * scale)
        ax.set_xlabel('Частота (Гц)', fontsize=font_size)
        ax.set_ylabel('Когерентність', fontsize=font_size)
        ax.set_title(f'Когерентність між {name1} та {name2}', fontsize=font_size + 2)
        ax.grid(True)
        ax.tick_params(labelsize=font_size * 0.9)
        fig.tight_layout()
        return fig

    elif mode == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=f,
            y=Cxy,
            mode='lines',
            line=dict(color='blue'),
            name='Coherence'
        ))

        fig.update_layout(
            title=f'Когерентність між {name1} та {name2}',
            xaxis_title='Частота (Гц)',
            yaxis_title='Когерентність',
            yaxis_type='log',
            height=400,
            width=800,
            margin=dict(t=40, b=40, l=60, r=40),
        )
        
        fig.update_yaxes(type="log", title_text="Когерентність", tickformat=".2f")
        
        
        
        return fig
    
# def coherent_subtraction_aligned_with_mask(sig_primary, 
#                                            sig_reference, fs=800,
#                                            seg_len_s=None, overlap_s=None,
#                                            coherence_threshold=0.7):
#     """
#     Когерентне віднімання з урахуванням маски когерентності.
#     Враховує затримку, фазу, амплітуду і обмежує віднімання тільки до когерентних частот.
#     """

#     # --- Внутрішні допоміжні функції ---
#     def estimate_delay(sig1, sig2):
#         """Оцінка затримки між сигналами через крос-кореляцію."""
#         corr = np.correlate(sig1, sig2, mode='full')
#         lag = np.argmax(corr) - len(sig2) + 1
#         return lag

#     def shift_signal(sig, delay):
#         """Зсув сигналу на задану кількість семплів."""
#         if delay > 0:
#             return np.pad(sig, (delay, 0), mode='constant')[:len(sig)]
#         elif delay < 0:
#             return np.pad(sig, (0, -delay), mode='constant')[-delay:]
#         else:
#             return sig

#     def normalize_signal(sig):
#         """Нормалізація сигналу за амплітудою."""
#         return sig / (np.max(np.abs(sig)) + 1e-10)


#     x = sig_primary.values


#     # Обчислюємо nperseg та noverlap
#     if seg_len_s is not None:
#         nperseg = int(seg_len_s * fs)
#     else:
#         nperseg = min(int(fs * 2), len(x))
#         if len(x) == nperseg:
#             nperseg = int(len(x) / 6)

#     if overlap_s is not None:
#         noverlap = int(overlap_s * fs)
#     else:
#         noverlap = int(nperseg * 0.9)

#     # Захист від помилки: noverlap must be less than nperseg
#     if noverlap >= nperseg:
#         noverlap = nperseg - 1

#     print('coherence')
#     print(nperseg)
#     print(noverlap)



#     # --- 1. Оцінка затримки ---
#     delay = estimate_delay(sig_primary, sig_reference)

#     # --- 2. Вирівнювання сигналів ---
#     sig_ref_aligned = shift_signal(sig_reference, delay)

#     # --- 3. STFT ---
#     f, t_stft, Z_primary = stft(sig_primary, fs=fs, nperseg=nperseg, noverlap=noverlap)
#     _, _, Z_reference = stft(sig_ref_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

#     # --- 4. Оцінка когерентності ---
#     f_coh, Cxy = coherence(sig_primary, sig_ref_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

#     # Побудова маски когерентності
#     coh_mask = (Cxy > coherence_threshold).astype(float)  # 1 – когерентна, 0 – ні
#     coh_mask_2d = coh_mask[:, np.newaxis]  # для STFT (freq x time)

#     # --- 5. Когерентне віднімання з маскою ---
#     gain = np.abs(Z_primary) / (np.abs(Z_reference) + 1e-10)
#     phase_correction = Z_reference / (np.abs(Z_reference) + 1e-10)
#     noise_estimate = gain * phase_correction

#     Z_clean = Z_primary - coh_mask_2d * noise_estimate  # застосування маски

#     # --- 6. Обернене перетворення ---
#     _, signal_cleaned = istft(Z_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)

#     # --- 7. Постобробка ---
#     signal_cleaned = signal_cleaned[:len(sig_primary)]
#     signal_cleaned = normalize_signal(signal_cleaned)

#     return signal_cleaned, delay, coh_mask, f_coh

from scipy.signal import stft, istft, correlate, windows
from scipy.signal.windows import hann

def coherent_subtraction_aligned_with_mask(sig_primary,
                                           sig_reference, fs=800,
                                           seg_len_s=None, overlap_s=None,
                                           coherence_threshold=0.7):
    def estimate_delay(sig1, sig2):
        n = len(sig1)
        win = windows.hann(n)
        corr = correlate(sig1 * win, sig2 * win, mode='full')
        lags = np.arange(-n + 1, n)
        delay = lags[np.argmax(corr)]
        return delay

    def shift_signal(sig, delay):
        if delay > 0:
            return np.pad(sig, (delay, 0), mode='constant')[:len(sig)]
        elif delay < 0:
            return np.pad(sig, (0, -delay), mode='constant')[-delay:]
        else:
            return sig

    def normalize_signal(sig):
        return sig / (np.max(np.abs(sig)) + 1e-10)

    x = sig_primary
    y = sig_reference

    nperseg = int(seg_len_s * fs) if seg_len_s else min(int(fs * 2), len(x))
    noverlap = int(overlap_s * fs) if overlap_s else int(nperseg * 0.9)
    noverlap = min(noverlap, nperseg - 1)

    delay = estimate_delay(x, y)
    y_aligned = shift_signal(y, delay)

    f, t_stft, Zx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zy = stft(y_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

    Zxy = Zx * np.conj(Zy)
    Sxx = np.abs(Zx) ** 2
    Syy = np.abs(Zy) ** 2
    coherence_local = np.abs(Zxy) ** 2 / (Sxx * Syy + 1e-10)

    coh_mask = (coherence_local > coherence_threshold).astype(float)

    snr_estimate = Sxx / (Syy + 1e-10)
    wiener_mask = snr_estimate / (snr_estimate + 1.0)

    suppression_mask = coh_mask * (1.0 - wiener_mask)

    Z_clean = Zx * (1.0 - suppression_mask)

    _, signal_cleaned = istft(Z_clean, fs=fs, nperseg=nperseg, noverlap=noverlap)
    signal_cleaned = signal_cleaned[:len(x)]
    # signal_cleaned = normalize_signal(signal_cleaned)

    return signal_cleaned, delay, coherence_local, f, t_stft, Zx, Z_clean



def smooth_dataframe(df, method='savgol', fs=1000, **kwargs):
    """
    Застосовує згладжування до кожної колонки DataFrame з урахуванням крайових ефектів.

    Parameters:
    - df: pandas.DataFrame — вхідний датафрейм із сигналами
    - method: str — метод згладжування ('savgol', 'moving', 'lowess')
    - fs: float — частота дискретизації (Гц)
    - **kwargs:
        - window_s: ширина вікна у секундах (альтернатива window_length/window)
        - polyorder: для 'savgol'
        - frac: для 'lowess'

    Returns:
    - pandas.DataFrame — датафрейм зі згладженими сигналами
    """
    df_smoothed = pd.DataFrame(index=df.index)

    for col in df.columns:
        signal = df[col].values
        n = len(signal)

        if method == 'savgol':
            if 'window_s' in kwargs:
                window_length = int(kwargs['window_s'] * fs)
                if window_length % 2 == 0:
                    window_length += 1
            else:
                window_length = kwargs.get('window_length', 11)

            polyorder = kwargs.get('polyorder', 2)

            # Дзеркальне доповнення
            pad = window_length // 2
            padded_signal = np.pad(signal, pad_width=pad, mode='reflect')
            smoothed_padded = savgol_filter(padded_signal, window_length=window_length, polyorder=polyorder)
            smoothed = smoothed_padded[pad:-pad]

        elif method == 'moving':
            if 'window_s' in kwargs:
                window = int(kwargs['window_s'] * fs)
            else:
                window = kwargs.get('window', 5)

            pad = window // 2
            padded_signal = np.pad(signal, pad_width=pad, mode='reflect')
            smoothed_padded = pd.Series(padded_signal).rolling(window=window, center=True).mean().to_numpy()
            smoothed = smoothed_padded[pad:-pad]

        elif method == 'lowess':
            # LOWESS сам обробляє краї за допомогою ваг
            frac = kwargs.get('frac', 0.1)
            smoothed = lowess(signal, np.arange(n), frac=frac, return_sorted=False)

        else:
            raise ValueError(f"Метод '{method}' не підтримується. Оберіть 'savgol', 'moving', або 'lowess'.")

        df_smoothed[col] = smoothed

    return df_smoothed


def compute_rms(signal, fs, start_time, end_time):
    """
    Обчислює RMS сигналу у заданому часовому діапазоні.

    Parameters:
        signal (np.ndarray): масив сигналу.
        fs (float): частота дискретизації (Гц).
        start_time (float): початок вікна (секунди).
        end_time (float): кінець вікна (секунди).

    Returns:
        float: RMS значення у вікні.
    """
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    windowed_signal = signal[start_idx:end_idx]
    rms = np.sqrt(np.mean(windowed_signal**2))
    return rms

def rms_in_band(freqs, signal, min_freq, max_freq):
    """
    Обчислює середнє квадратичне значення сигналу в заданому частотному діапазоні.

    Parameters:
        freqs (np.ndarray): масив частот (1D).
        signal (np.ndarray): спектральний сигнал (1D), такий самий розмір як freqs.
        min_freq (float): мінімальна частота діапазону.
        max_freq (float): максимальна частота діапазону.

    Returns:
        rms_value (float): середнє квадратичне значення у вибраному діапазоні.
        band_freqs (tuple): фактичні частоти меж діапазону (f_start, f_end).
    """

    # Знаходимо індекси найближчих значень у масиві частот
    idx_start = np.argmin(np.abs(freqs - min_freq))
    idx_end = np.argmin(np.abs(freqs - max_freq))

    # Сортуємо індекси на випадок, якщо max_freq < min_freq
    idx_min, idx_max = sorted([idx_start, idx_end])

    # Вирізаємо діапазон частот та сигналу
    selected_signal = signal[idx_min:idx_max + 1]

    # Обчислюємо RMS
    rms_value = np.sqrt(np.mean(selected_signal**2))
    return rms_value, (freqs[idx_min], freqs[idx_max])


def vpf_df(df, fs, columns=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3']):
    """
    Виконує векторну поляризаційну фільтрацію (VPF) та візуалізує уявну частину комплексної потужності.

    Parameters:
        Vr (np.ndarray): радіальна компонента сигналу.
        Vz (np.ndarray): вертикальна компонента сигналу.
        fs (float): частота дискретизації (Гц).
        mode (str): 'matrix' — повертає масив; 'fig' — повертає matplotlib.figure; 'plotly' — повертає plotly.graph_objects.Figure.
        scale (float): масштаб (тільки для matplotlib), за замовчуванням 1.0.

    Returns:
        або: np.ndarray (imag_VPP), або matplotlib.figure.Figure, або plotly.graph_objects.Figure
    """
    n_samples = len(df)
    time = np.arange(n_samples) / fs

    # Обчислення аналітичних сигналів
    Vr1 = []
    Vr2 = []
    Vr3 = []
    Vz1 = []
    Vz2 = []
    Vz3 = []
    
    VrVz_dict = {}
    
    if all(elem in list(df.columns) for elem in columns):
    
        Vr1.append(df['X1'])
        Vr2.append(df['X2'])
        Vr3.append(df['X3'])
        Vz1.append(-df['Z1'])
        Vz2.append(-df['Z2'])
        Vz3.append(-df['Z3'])
        Vr = {'1':Vr1, '2':Vr2, '3':Vr3}
        Vz = {'1':Vz1, '2':Vz2, '3':Vz3}
        VrVz_dict['Vr'] = Vr
        VrVz_dict['Vz'] = Vz
        
    print('####################   VPF  ###########################')
    print(VrVz_dict['Vr']['1'])
    # breakpoint()
        
    im_power1 = vpf(np.array(VrVz_dict['Vr']['1']), np.array(VrVz_dict['Vz']['1']), fs, mode='matrix') 
    im_power2 = vpf(np.array(VrVz_dict['Vr']['2']), np.array(VrVz_dict['Vz']['2']), fs, mode='matrix')
    im_power3 = vpf(np.array(VrVz_dict['Vr']['3']), np.array(VrVz_dict['Vz']['3']), fs, mode='matrix') 

    print(im_power1)
    # breakpoint()

    im_power_df = pd.DataFrame({'im_power1':im_power1[0],'im_power2':im_power2[0], 'im_power3':im_power3[0]})
    
    return im_power_df
       

    


