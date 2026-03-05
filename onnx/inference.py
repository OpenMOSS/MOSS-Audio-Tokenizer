"""
ONNX Runtime backend for MOSS Audio Tokenizer.

Drop-in replacement for the PyTorch ``MossAudioTokenizerModel`` that runs
encode / decode through ONNX Runtime — no PyTorch dependency required.

Supports CUDA, TensorRT, and CPU execution providers via ORT's provider list.

Model I/O (identical to PyTorch version):
  Encoder: input_values (1,1,T) float32, n_quantizers () int64
           → audio_codes (32,1,T') int64, audio_codes_lengths (1,) int64
  Decoder: audio_codes (32,1,T') int64, n_quantizers () int64
           → audio (1,1,T) float32, audio_lengths (1,) int64
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DOWNSAMPLE_RATE = 1920
SAMPLE_RATE = 24000
N_QUANTIZERS = 32


def _load_ort_session(onnx_path: str | Path, use_gpu: bool = True):
    import onnxruntime as ort

    providers: list[str] = []
    if use_gpu:
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    log.info("Loaded %s — providers=%s", Path(onnx_path).name, session.get_providers())
    return session


class OnnxAudioTokenizer:
    """Encode waveforms → audio codes and decode codes → waveforms via ONNX Runtime.

    Args:
        encoder_path: path to encoder ONNX model
        decoder_path: path to decoder ONNX model
        n_quantizers: number of RVQ codebooks (default 32)
        use_gpu: prefer CUDA / TensorRT execution providers
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        n_quantizers: int = N_QUANTIZERS,
        use_gpu: bool = True,
    ):
        self.n_quantizers = n_quantizers
        self.sample_rate = SAMPLE_RATE

        encoder_path = Path(encoder_path)
        decoder_path = Path(decoder_path)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        self._encoder = _load_ort_session(encoder_path, use_gpu)
        self._decoder = _load_ort_session(decoder_path, use_gpu)

        self._enc_in = [i.name for i in self._encoder.get_inputs()]
        self._enc_out = [o.name for o in self._encoder.get_outputs()]
        self._dec_in = [i.name for i in self._decoder.get_inputs()]
        self._dec_out = [o.name for o in self._decoder.get_outputs()]

        log.info(
            "OnnxAudioTokenizer ready: n_quantizers=%d, sr=%d, downsample=%d",
            n_quantizers, SAMPLE_RATE, DOWNSAMPLE_RATE,
        )

    def encode(self, waveform: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Encode a waveform to audio codes.

        Args:
            waveform: float32 array, shape ``(T,)`` or ``(1, T)`` or ``(1, 1, T)``
            n_quantizers: RVQ layers to use (default: ``self.n_quantizers``)

        Returns:
            audio_codes: int64 array, shape ``(T', n_vq)``
        """
        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]

        T = waveform.shape[-1]
        padded = ((T + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != T:
            waveform = np.concatenate(
                [waveform, np.zeros((1, 1, padded - T), dtype=np.float32)], axis=-1,
            )

        waveform = waveform.astype(np.float32)
        nq = np.array(n_quantizers, dtype=np.int64)

        outputs = self._encoder.run(
            self._enc_out,
            {self._enc_in[0]: waveform, self._enc_in[1]: nq},
        )
        audio_codes = outputs[0]       # (32, 1, T')
        code_lengths = outputs[1]      # (1,)

        code_len = int(code_lengths[0])
        codes = audio_codes[:, 0, :code_len]  # (32, T')
        return codes.T.astype(np.int64)        # (T', 32)

    def decode(self, audio_codes: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Decode audio codes to a waveform.

        Args:
            audio_codes: int64 array, shape ``(T', n_vq)`` or ``(n_vq, T')``
            n_quantizers: RVQ layers (default: ``self.n_quantizers``)

        Returns:
            waveform: float32 array, shape ``(T,)``
        """
        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if audio_codes.ndim == 2:
            if audio_codes.shape[1] == self.n_quantizers and audio_codes.shape[0] != self.n_quantizers:
                audio_codes = audio_codes.T
            codes_3d = audio_codes[:, np.newaxis, :]
        elif audio_codes.ndim == 3:
            codes_3d = audio_codes
        else:
            raise ValueError(f"Expected 2D or 3D codes, got {audio_codes.ndim}D")

        codes_3d = codes_3d.astype(np.int64)
        nq = np.array(n_quantizers, dtype=np.int64)

        outputs = self._decoder.run(
            self._dec_out,
            {self._dec_in[0]: codes_3d, self._dec_in[1]: nq},
        )
        audio = outputs[0]          # (1, 1, T)
        audio_lengths = outputs[1]  # (1,)

        length = int(audio_lengths[0])
        return audio[0, 0, :length].astype(np.float32)
