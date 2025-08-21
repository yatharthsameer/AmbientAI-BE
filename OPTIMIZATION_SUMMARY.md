# CPU-Only MVP Optimization Summary

This document summarizes the high-value fixes implemented to achieve a better **accuracy ⇄ latency** balance for CPU-only transcription.

## ✅ Implemented Fixes

### 1. Fixed VAD Windowing (HUGE accuracy win)
- **Problem**: VAD was dropping all silences, creating "Franken-speech" windows that destroyed prosody/context
- **Solution**: 
  - Use VAD only to choose cut points, not to remove internal silences
  - Build windows directly from raw PCM and preserve natural silences
  - Live windows: **25s** length, **2s overlap**
  - Final windows: **30s** length, **2s overlap**

### 2. Enabled Model's Own VAD + Word Timestamps
- **Problem**: `vad_filter=False` and missing `word_timestamps` led to poor segmentation
- **Solution**:
  - `vad_filter=True` with `vad_parameters={"min_silence_duration_ms":600,"speech_pad_ms":150}`
  - `word_timestamps=True` for timestamp-based deduplication
  - Enables clean stitching by timestamps in overlaps

### 3. Reduced Live Latency
- **Problem**: Live pass waited for 60s macro blocks with heavy beam search
- **Solution**:
  - Start decoding at **25s** windows (reduced from 60s)
  - Use **`beam_size=3`** (reduced from 6), `patience=1.0`, `temperature=0.0`
  - Keep `chunk_length=25`, `language="en"`

### 4. Implemented Prompt Discipline
- **Problem**: Concatenating ~1000 chars of prior text with `condition_on_previous_text=False` was wasteful
- **Solution**:
  - **Live**: Short, static SOC/OASIS prompt only (no rolling context)
  - **Final**: `condition_on_previous_text=True` with last 1-2 sentences (≤600 chars)

### 5. Optimized CPU-Only Model Choices
- **Problem**: `int8` everywhere dented accuracy; `large-v3` on CPU live was too slow
- **Solution**:
  - **Live**: `medium.en` + `compute_type="int8_float16"` (CPU optimized)
  - **Final**: `large-v3` + `compute_type="int8_float16"` (high accuracy, offline)

### 6. High-Quality Single Resample
- **Problem**: Mixed resamplers (pydub/librosa) + temp WAV churn
- **Solution**:
  - Keep master at **48kHz**; resample once to **16kHz mono** with **soxr (VHQ)**
  - Feed **numpy float32 arrays** to faster-whisper (avoid temp files)

### 7. Timestamp-Based Stitching
- **Problem**: `_join_with_dedup` trimmed by textual prefixes (brittle)
- **Solution**: 
  - In 2s overlap, drop earlier words whose timestamps fall inside later window's range
  - Uses `word_timestamps=True` from fix #2

### 8. Safer Post-Processing
- **Problem**: `_clean_domain_text` globally turned " over " → "/" (could corrupt content)
- **Solution**: 
  - Scoped rules: only replace " over " when flanked by digits (`\d+\s+over\s+\d+`)
  - Keep raw ASR for evidence

### 9. Concurrency Guard at Session End
- **Problem**: Starting "flush macro" and "full refine" together caused CPU contention
- **Solution**: 
  - On `end_session`: **await macro flush → then run final pass** (single worker per session)

### 10. Minimal Thresholds on Live
- **Solution**: Added thresholds to cut unstable segments early:
  - `no_speech_threshold = 0.6`
  - `compression_ratio_threshold = 2.3`
  - `log_prob_threshold = -0.5`

## 📊 Parameter Presets

### Live/Provisional (CPU-Optimized)
```python
{
    "model": "medium.en",
    "compute_type": "int8_float16",
    "chunk_length": 25,
    "overlap": 2,
    "beam_size": 3,
    "patience": 1.0,
    "temperature": 0.0,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 600, "speech_pad_ms": 150},
    "word_timestamps": True,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.3,
    "logprob_threshold": -0.5,
    "condition_on_previous_text": False,
    "prompt": "Short static SOC/OASIS only"
}
```

### Final/Offline (High-Accuracy)
```python
{
    "model": "large-v3",
    "compute_type": "int8_float16",
    "chunk_length": 30,
    "overlap": 2,
    "beam_size": 8,
    "patience": 1.1,
    "temperature": [0.0, 0.2, 0.4],
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 600, "speech_pad_ms": 150},
    "word_timestamps": True,
    "condition_on_previous_text": True,
    "prompt": "SOC/OASIS + last 1-2 sentences (≤600 chars)"
}
```

## 🔧 Technical Changes

### Configuration Updates (`config.py`)
- Updated WebSocket settings for 25s chunks with 2s overlap
- Added separate models for live (`medium.en`) and final (`large-v3`) processing
- Changed compute type to `int8_float16` for better CPU performance

### Transcription Service (`transcription.py`)
- Added high-quality soxr resampling with VHQ quality
- Implemented direct numpy array processing for faster-whisper
- Added safer post-processing with scoped regex rules
- Enhanced word-level timestamp extraction
- Added new parameters for VAD, thresholds, and word timestamps

### WebSocket Transcription (`websocket_transcription.py`)
- Updated live transcription parameters for CPU optimization
- Added timestamp-based deduplication method
- Implemented optimized beam search and thresholds

### Minimal Server (`minimal_server.py`)
- Reduced macro block size from 60s to 25s
- Updated window sizes (25s-30s with 2s overlap)
- Implemented concurrency guard for session end processing
- Added safer post-processing with scoped rules
- Updated prompt discipline for live vs final passes

### Dependencies (`requirements.txt`)
- Added `soxr` for high-quality audio resampling

## 🎯 Expected Results

These optimizations should deliver:

1. **Materially improved provisional accuracy** by preserving natural speech prosody
2. **Reduced live latency** through smaller decode windows and lighter beam search
3. **Better CPU utilization** with optimized model choices and compute types
4. **Cleaner stitching** through timestamp-based deduplication
5. **Safer post-processing** that doesn't corrupt medical content
6. **Reduced garbage output** through appropriate thresholds

The system now provides a good balance between real-time performance and accuracy, with a high-quality final pass for the permanent record.
