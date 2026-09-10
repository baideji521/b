"""音频 DSP 内核：STFT / mel / MFCC / chroma / onset / tempo / beat_track。

**全部用 numpy 手写，不引入 librosa 或 scipy。** 理由写在 `dance/__init__.py` 里：
librosa 拖着 numba + llvmlite 一整条编译链，而这里真正要的东西加起来只有几百行，
项目 requirements 已有 `numpy>=1.26`，不必为此加依赖（技术指导第一节第 18 条：
依赖不适合当前环境就不要强行引入）。

算法出处（只吸收思想，代码在此重新实现）：
- 谱通量 onset envelope：Dixon 2006 的经典做法，半波整流后按频带求和
- tempo 自相关 + 倍频合并：避免 128 BPM 的曲子被判成 64 BPM
- beat_track 动态规划：Ellis 2007《Beat Tracking by Dynamic Programming》的
  cumscore/backlink 形式，转移代价 -tightness * log(v/period)^2

时间口径：第 i 帧的中心时刻是 `i * hop / sample_rate`（不做 center padding，
所以第 0 帧对应 0 秒，和 `video_io` 那边"帧号除以 fps"的口径一致）。
"""

from __future__ import annotations

import numpy as np

#: 默认分析参数。改这些会让 alignment / 音乐结构的缓存全部失效，所以进 config_hash
DEFAULT_SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 64
FMIN = 20.0
#: 分块 rfft 的块大小：一次铺开 (帧数 x n_fft) 的矩阵会吃掉几百 MB
_BLOCK = 256
#: beat_track 的转移代价系数。越大越倾向严格等间距（Ellis 原文用 100）
TIGHTNESS = 100.0
#: tempo 搜索范围。卡点舞素材基本落在这个区间
BPM_MIN = 60.0
BPM_MAX = 190.0
#: tempo 对数正态先验的中心与宽度（宽度单位是八度）。见 `estimate_tempo` 的说明
PRIOR_BPM = 120.0
PRIOR_WIDTH = 0.8


_EPS = 1e-10


def hann(size: int) -> np.ndarray:
    """周期性 Hann 窗（和 STFT 重叠相加的习惯一致）。"""
    n = np.arange(size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)).astype(np.float32)


def frame_count(length: int, n_fft: int = N_FFT, hop: int = HOP) -> int:
    if length < n_fft:
        return 0
    return 1 + (length - n_fft) // hop


def frame_times(frames: int, sample_rate: int = DEFAULT_SR, hop: int = HOP) -> np.ndarray:
    """每帧的中心时刻（秒）。不做 center padding，第 0 帧就是 0 秒。"""
    return (np.arange(frames, dtype=np.float64) * hop / float(sample_rate))


def stft_magnitude(pcm: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """幅度谱，形状 `(帧数, n_fft // 2 + 1)`，float32。

    分块做 rfft：3 分钟 22050Hz 的曲子有约 7700 帧，一次铺开 7700x2048 的 float32
    就是 63MB，再乘上 rfft 的复数中间结果会更多，分块之后峰值只有几 MB。
    """
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    frames = frame_count(pcm.size, n_fft, hop)
    bins = n_fft // 2 + 1
    if frames <= 0:
        return np.zeros((0, bins), dtype=np.float32)
    window = hann(n_fft)
    out = np.empty((frames, bins), dtype=np.float32)
    block = np.empty((min(_BLOCK, frames), n_fft), dtype=np.float32)
    for begin in range(0, frames, _BLOCK):
        count = min(_BLOCK, frames - begin)
        for row in range(count):
            offset = (begin + row) * hop
            block[row] = pcm[offset:offset + n_fft]
        spec = np.fft.rfft(block[:count] * window, axis=1)
        out[begin:begin + count] = np.abs(spec).astype(np.float32)
    return out


def fft_frequencies(sample_rate: int = DEFAULT_SR, n_fft: int = N_FFT) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK 口径的 mel 刻度（和 mel_to_hz 严格互逆）。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = DEFAULT_SR, n_fft: int = N_FFT,
                   n_mels: int = N_MELS, fmin: float = FMIN,
                   fmax: float | None = None) -> np.ndarray:
    """三角 mel 滤波器组，形状 `(n_mels, n_fft // 2 + 1)`，各带峰值归一到 1。

    峰值归一（而不是面积归一）是有意的：后面只用它算"哪个频带变强了"，
    面积归一会让高频带因为更宽而被压得几乎看不见。
    """
    top = float(fmax) if fmax else sample_rate / 2.0
    freqs = fft_frequencies(sample_rate, n_fft)
    points = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(top), n_mels + 2))
    bank = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = points[i], points[i + 1], points[i + 2]
        if right - left <= 0:
            continue
        rise = (freqs - left) / max(center - left, _EPS)
        fall = (right - freqs) / max(right - center, _EPS)
        bank[i] = np.maximum(0.0, np.minimum(rise, fall)).astype(np.float32)
    peaks = bank.max(axis=1, keepdims=True)
    np.divide(bank, np.maximum(peaks, _EPS), out=bank)
    return bank


def melspectrogram(pcm: np.ndarray, sample_rate: int = DEFAULT_SR, n_fft: int = N_FFT,
                   hop: int = HOP, n_mels: int = N_MELS) -> np.ndarray:
    """功率 mel 谱，形状 `(帧数, n_mels)`。"""
    mag = stft_magnitude(pcm, n_fft, hop)
    if mag.shape[0] == 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    bank = mel_filterbank(sample_rate, n_fft, n_mels)
    return (mag ** 2) @ bank.T


def power_to_db(power: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """功率转 dB，并把动态范围压到 top_db（避免静音段的 -inf 主导后续统计）。"""
    power = np.asarray(power, dtype=np.float32)
    if power.size == 0:
        return power
    db = 10.0 * np.log10(np.maximum(power, _EPS))
    return np.maximum(db, db.max() - float(top_db)).astype(np.float32)


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """正交 DCT-II 矩阵，形状 `(n_out, n_in)`。MFCC 就是 log-mel 乘它。"""
    k = np.arange(n_out, dtype=np.float64)[:, None]
    n = np.arange(n_in, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in))
    basis *= np.sqrt(2.0 / n_in)
    basis[0] *= np.sqrt(0.5)
    return basis.astype(np.float32)


def mfcc(pcm: np.ndarray, sample_rate: int = DEFAULT_SR, n_mfcc: int = 13,
         n_fft: int = N_FFT, hop: int = HOP, n_mels: int = N_MELS) -> np.ndarray:
    """MFCC，形状 `(帧数, n_mfcc)`。用于音乐段落聚类的音色维度。"""
    mel = melspectrogram(pcm, sample_rate, n_fft, hop, n_mels)
    if mel.shape[0] == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    return power_to_db(mel) @ dct_matrix(n_mfcc, mel.shape[1]).T


def chromagram(pcm: np.ndarray, sample_rate: int = DEFAULT_SR, n_fft: int = N_FFT,
               hop: int = HOP, fmin: float = 55.0, fmax: float = 2093.0) -> np.ndarray:
    """12 维 chroma，形状 `(帧数, 12)`，每帧 L2 归一。

    做法是把每个 FFT bin 按 `12 * log2(f / 440) + 69` 折算成 MIDI 音高、再取模 12
    归到音级。这不是 CQT，但对"两段音频是不是同一首歌"这个用途足够：
    我们只需要一条对移调/音量不敏感、对和声走向敏感的特征序列。
    频率范围掐在 A1~C7：更低的基频分辨率不够，更高的基本是打击乐噪声。
    """
    mag = stft_magnitude(pcm, n_fft, hop)
    if mag.shape[0] == 0:
        return np.zeros((0, 12), dtype=np.float32)
    freqs = fft_frequencies(sample_rate, n_fft)
    usable = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not usable.any():
        return np.zeros((mag.shape[0], 12), dtype=np.float32)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs[usable], _EPS) / 440.0)
    classes = np.mod(np.round(midi).astype(np.int64), 12)
    # 一次矩阵乘搞定所有帧：投影矩阵 (bins x 12)，命中的 bin 置 1
    project = np.zeros((int(usable.sum()), 12), dtype=np.float32)
    project[np.arange(classes.size), classes] = 1.0
    chroma = mag[:, usable] @ project
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    return (chroma / np.maximum(norm, _EPS)).astype(np.float32)


def smooth(track: np.ndarray, width: int = 5) -> np.ndarray:
    """移动平均，边界用边缘值补齐（不引入零，避免首尾被人为压低）。"""
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    width = int(width)
    if width <= 1 or track.size == 0:
        return track
    pad = width // 2
    padded = np.pad(track, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(padded, kernel, mode="valid")[:track.size].astype(np.float32)


def normalize01(track: np.ndarray) -> np.ndarray:
    """线性拉到 0~1。全平（max == min）时返回全 0，不返回 NaN。"""
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return track
    lo, hi = float(track.min()), float(track.max())
    if hi - lo <= _EPS:
        return np.zeros_like(track)
    return ((track - lo) / (hi - lo)).astype(np.float32)


def robust_normalize01(track: np.ndarray, low: float = 2.0,
                       high: float = 98.0) -> np.ndarray:
    """按分位数裁剪后再拉到 0~1。给**特征曲线**用，不给 onset 包络用。

    和 `normalize01` 的区别只有一件事：用 2/98 分位代替最小值/最大值。
    为什么要这样：min-max 会被单个瞬态毁掉 —— 一首歌里有一下爆音，
    整条能量曲线就被压到 0.1 以下，"这个音乐位置有多带劲"全变成 0，
    评分里的 `position_match` 也就废了。裁掉两端 2% 之后曲线才有动态范围。

    **不能**拿它去归一化 onset 包络：节拍跟踪要的正是那些尖峰，
    裁掉最高的 2% 等于把最强的鼓点削平，反而更难跟上拍。
    """
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return track
    lo = float(np.percentile(track, max(0.0, float(low))))
    hi = float(np.percentile(track, min(100.0, float(high))))
    if hi - lo <= _EPS:
        return normalize01(track)
    return np.clip((track - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)



def onset_strength(mel_db: np.ndarray, smooth_width: int = 3) -> np.ndarray:
    """谱通量 onset 包络，长度等于帧数，已归一到 0~1。

    只取"变强"的部分（半波整流）：鼓点落下时几十个频带同时上冲，会顶出一个尖峰；
    声音衰减时不产生峰，所以一个鼓点只被算一次。第 0 帧固定为 0（没有前一帧可比）。
    """
    mel_db = np.asarray(mel_db, dtype=np.float32)
    if mel_db.ndim != 2 or mel_db.shape[0] < 2:
        return np.zeros(max(0, mel_db.shape[0] if mel_db.ndim == 2 else 0), dtype=np.float32)
    diff = np.diff(mel_db, axis=0)
    np.maximum(diff, 0.0, out=diff)
    env = np.concatenate([[0.0], diff.sum(axis=1)]).astype(np.float32)
    return normalize01(smooth(env, smooth_width))


def onset_envelope(pcm: np.ndarray, sample_rate: int = DEFAULT_SR, n_fft: int = N_FFT,
                   hop: int = HOP, n_mels: int = N_MELS) -> tuple[np.ndarray, float]:
    """从波形直接算 onset 包络，返回 `(包络, 包络帧率)`。"""
    mel = melspectrogram(pcm, sample_rate, n_fft, hop, n_mels)
    env = onset_strength(power_to_db(mel))
    return env, float(sample_rate) / float(hop)


def autocorrelation(track: np.ndarray) -> np.ndarray:
    """去均值后的自相关（只要非负 lag），用 FFT 算，O(n log n)。"""
    track = np.asarray(track, dtype=np.float64).reshape(-1)
    if track.size < 2:
        return np.zeros(track.size, dtype=np.float64)
    centered = track - track.mean()
    size = 1 << int(np.ceil(np.log2(centered.size * 2)))
    spectrum = np.fft.rfft(centered, size)
    return np.fft.irfft(spectrum * np.conj(spectrum), size)[:centered.size].real


def estimate_tempo(env: np.ndarray, env_rate: float, bpm_min: float = BPM_MIN,
                   bpm_max: float = BPM_MAX, prior_bpm: float = PRIOR_BPM,
                   prior_width: float = PRIOR_WIDTH) -> tuple[float, float]:
    """自相关求 BPM，返回 `(bpm, 置信度 0~1)`。算不出来返回 `(0.0, 0.0)`。

    三件事叠在一起才靠得住，少任何一件都会出现"倍速/半速"的经典错判：

    1. **去锥形偏置**：FFT 自相关在 lag 处只有 `n - lag` 项重叠，原始值随 lag 线性衰减，
       直接取峰会系统性偏向短 lag（快速）。除以重叠项数还原成无偏估计。
    2. **对数正态 tempo 先验**：以 `prior_bpm`（120）为中心、在 log2 域按 `prior_width`
       个八度衰减。人对 60/120/240 这一列同样"合拍"的候选里天然偏好 120 附近，
       没有先验时纯凭自相关无法在数学上区分它们。
    3. **倍频加成**：`lag` 的得分再加上 `2*lag` 的一半。真实拍点在 2 倍 lag 处也有峰，
       这一项让"确实是拍"的候选比"恰好碰上一个响声"的候选更突出。

    置信度 = 最佳得分相对全窗平均的突出程度，压到 0~1。
    """
    env = np.asarray(env, dtype=np.float32).reshape(-1)
    if env.size < 16 or float(env.max(initial=0.0)) <= 0.0:
        return 0.0, 0.0
    ac = autocorrelation(env)
    counts = np.maximum(env.size - np.arange(ac.size), 1.0)
    ac = ac / counts                                    # 1. 去锥形偏置
    lag_min = max(1, int(round(env_rate * 60.0 / max(bpm_max, 1.0))))
    lag_max = min(ac.size - 1, int(round(env_rate * 60.0 / max(bpm_min, 1.0))))
    if lag_max <= lag_min:
        return 0.0, 0.0
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * env_rate / lags
    score = ac[lags].astype(np.float64, copy=True)
    doubled = lags * 2
    inside = doubled < ac.size
    score[inside] += 0.5 * ac[doubled[inside]]          # 3. 倍频加成
    prior = np.exp(-0.5 * (np.log2(bpms / float(prior_bpm)) / float(prior_width)) ** 2)
    score = score * prior                               # 2. tempo 先验
    best = int(np.argmax(score))
    peak = float(score[best])
    mean = float(np.mean(np.abs(score))) or _EPS
    confidence = float(np.clip((peak / mean - 1.0) / 3.0, 0.0, 1.0))
    return round(float(bpms[best]), 3), round(confidence, 4)



def beat_track(env: np.ndarray, env_rate: float, bpm: float,
               tightness: float = TIGHTNESS) -> np.ndarray:
    """动态规划节拍跟踪，返回节拍所在的**帧号**（升序）。

    Ellis 2007 的形式，在这里重新实现：

        cumscore[t] = localscore[t] + max_v ( cumscore[t+v] + txcost(v) )
        txcost(v)   = -tightness * ( log(-v / period) ) ^ 2      （v 为负偏移）

    也就是"这一帧的 onset 有多强"加上"离上一拍的间距有多接近理论拍长"，
    搜索窗只看 [t - 2*period, t - period/2]。最后从 cumscore 最大的尾巴回溯。

    纯等间距网格搞不定真实音乐（前奏空拍、渐快、切分），所以必须上 DP；
    但 DP 也不能没有 tempo 先验，否则会贴着每一个碎响走。
    """
    env = np.asarray(env, dtype=np.float64).reshape(-1)
    if env.size < 4 or bpm <= 0 or env_rate <= 0:
        return np.zeros(0, dtype=np.int64)
    period = env_rate * 60.0 / float(bpm)
    if period < 1.0 or period >= env.size:
        return np.zeros(0, dtype=np.int64)
    local = env - env.mean()
    local = local / max(float(np.std(local)), _EPS)
    lo = max(1, int(round(period * 0.5)))
    hi = max(lo + 1, int(round(period * 2.0)))
    offsets = np.arange(lo, hi + 1, dtype=np.int64)
    # 转移代价：offsets 是"往前多少帧"，代价按它与理论拍长的对数偏差平方算
    txcost = -tightness * (np.log(offsets / period) ** 2)
    cumscore = np.zeros(env.size, dtype=np.float64)
    backlink = np.full(env.size, -1, dtype=np.int64)
    cumscore[0] = local[0]
    for t in range(1, env.size):
        prev = t - offsets
        valid = prev >= 0
        if not valid.any():
            cumscore[t] = local[t]
            continue
        scores = cumscore[prev[valid]] + txcost[valid]
        pick = int(np.argmax(scores))
        cumscore[t] = local[t] + float(scores[pick])
        backlink[t] = int(prev[valid][pick])
    # 尾巴只在最后一个拍长里找：整条最大值常常落在结尾的响声上，那不是拍
    tail_from = max(0, env.size - int(round(period)) - 1)
    tail = tail_from + int(np.argmax(cumscore[tail_from:]))
    beats: list[int] = []
    cursor = tail
    while cursor >= 0:
        beats.append(cursor)
        cursor = int(backlink[cursor])
    beats.reverse()
    return np.asarray(beats, dtype=np.int64)


def pick_peaks(track: np.ndarray, min_distance: int = 1,
               threshold_ratio: float = 0.25) -> np.ndarray:
    """局部极大值峰值检测，返回帧号。

    `threshold_ratio` 是相对最大值的门槛（0.25 = 只要高于峰值 1/4 的才算），
    `min_distance` 是最小间隔帧数（去抖，防止一个鼓点报出连续两个峰）。
    beat_track 失败时靠它兜底出一份节拍。
    """
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    if track.size < 3:
        return np.zeros(0, dtype=np.int64)
    floor = float(track.max()) * float(threshold_ratio)
    inner = np.arange(1, track.size - 1)
    hits = inner[(track[inner] > track[inner - 1]) & (track[inner] >= track[inner + 1])
                 & (track[inner] >= floor)]
    if hits.size == 0 or min_distance <= 1:
        return hits.astype(np.int64)
    kept: list[int] = []
    for index in hits:
        if not kept or index - kept[-1] >= min_distance:
            kept.append(int(index))
        elif track[index] > track[kept[-1]]:
            kept[-1] = int(index)          # 挨太近时留更强的那个
    return np.asarray(kept, dtype=np.int64)


def band_energy(mag: np.ndarray, sample_rate: int, n_fft: int,
                low_hz: float, high_hz: float) -> np.ndarray:
    """某个频带的逐帧能量（未归一）。kick/bass/clap/hihat 四条带都走它。"""
    mag = np.asarray(mag, dtype=np.float32)
    if mag.ndim != 2 or mag.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    freqs = fft_frequencies(sample_rate, n_fft)
    picks = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not picks.any():
        return np.zeros(mag.shape[0], dtype=np.float32)
    return (mag[:, picks] ** 2).sum(axis=1).astype(np.float32)


def spectral_centroid(mag: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """频谱重心（Hz）。整帧无能量时该帧记 0，不产生 NaN。"""
    mag = np.asarray(mag, dtype=np.float32)
    if mag.ndim != 2 or mag.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    freqs = fft_frequencies(sample_rate, n_fft).astype(np.float32)
    weight = mag.sum(axis=1)
    centroid = (mag @ freqs) / np.maximum(weight, _EPS)
    centroid[weight <= _EPS] = 0.0
    return centroid.astype(np.float32)


def frame_rms(pcm: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """逐帧 RMS。直接在时域算，比从谱里反推更直观也更快。"""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    frames = frame_count(pcm.size, n_fft, hop)
    if frames <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        chunk = pcm[i * hop:i * hop + n_fft]
        out[i] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return out


def envelope(pcm: np.ndarray, buckets: int = 600) -> np.ndarray:
    """画波形用的缩略包络：把整条音轨分成 `buckets` 段，每段取峰值绝对值，再归一到 0~1。

    **只给显示用**，不参与任何判定 —— 所以刻意取峰值而不是 RMS：
    界面上要一眼看出"鼓点打在哪儿"，RMS 会把它抹平。
    整条音轨几百万点在 Qt 里逐点画是不可能的，先降到几百个桶再画。
    """
    track = np.abs(np.asarray(pcm, dtype=np.float32).reshape(-1))
    size = max(1, int(buckets))
    if track.size == 0:
        return np.zeros(size, dtype=np.float32)
    if track.size < size:                 # 太短就原样返回，不无中生有地插值
        peak = float(track.max())
        return (track / peak if peak > _EPS else track).astype(np.float32)
    edges = np.linspace(0, track.size, size + 1).astype(np.int64)
    out = np.maximum.reduceat(track, edges[:-1]).astype(np.float32)
    peak = float(out.max())
    return (out / peak).astype(np.float32) if peak > _EPS else out


def cross_correlate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """FFT 全互相关，返回 `(相关序列, 零延迟所在下标)`。

    定义：`corr[lag] = Σ_n a[n + lag] · b[n]`，其中 `lag = index - zero_index`。
    所以峰值 lag 的含义是「**b 的内容出现在 a 里、位置往后偏了 lag 个采样**」。

    对齐层就是这么用的：`cross_correlate(目标歌, 源视频音轨)` 的峰值 lag 除以采样率
    直接就是 offset，满足全系统口径 `source_time = target_time - offset`。

    两条都先去均值：不去均值时直流分量会在 lag=0 附近堆出一个假峰。
    时域 `np.correlate` 是 O(n·m)，40 秒 22050Hz 就是 8.8e5 x 8.8e5，跑不动。
    """

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.float64), 0
    a = a - a.mean()
    b = b - b.mean()
    length = a.size + b.size - 1
    size = 1 << int(np.ceil(np.log2(max(length, 2))))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    # irfft 的结果是循环相关：前 a.size 个是非负 lag，尾部 b.size-1 个是负 lag。
    # 拼成"负 lag 在前"的常规顺序，零延迟就落在 b.size - 1
    full = np.concatenate([corr[-(b.size - 1):], corr[:a.size]]) if b.size > 1 else corr[:a.size]
    return full, b.size - 1


def correlate_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """逐带互相关再求和，用于 chroma（12 带）这类多维特征序列。

    输入形状都是 `(帧数, 带数)`。每一带各自做一次一维互相关，然后把 12 条相关曲线
    加起来 —— 这比"把矩阵拉平成一维"正确得多：拉平会让相邻带的样本互相串台。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), 0
    bands = min(a.shape[1], b.shape[1])
    total: np.ndarray | None = None
    zero = 0
    for band in range(bands):
        corr, zero = cross_correlate(a[:, band], b[:, band])
        total = corr if total is None else total + corr
    return (total if total is not None else np.zeros(0, dtype=np.float64)), zero







