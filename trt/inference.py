"""
Native TensorRT backend for MOSS Audio Tokenizer.

Uses pre-built ``.engine`` files — no PyTorch dependency.
Requires ``tensorrt`` and ``cuda-python`` packages.

Build engines from ONNX with ``trtexec``::

    trtexec --onnx=encoder.onnx --saveEngine=encoder.engine --fp16
    trtexec --onnx=decoder.onnx --saveEngine=decoder.engine --fp16
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DOWNSAMPLE_RATE = 1920
SAMPLE_RATE = 24000
N_QUANTIZERS = 32


class _TensorInfo:
    """Mimics ``onnxruntime.NodeArg`` for a uniform ``info.name`` interface."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


def _trt_to_np_dtype(trt_dtype):
    import tensorrt as trt
    _MAP = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int64: np.int64,
        trt.int32: np.int32,
        trt.int8: np.int8,
        trt.bool: np.bool_,
    }
    return _MAP.get(trt_dtype, np.float32)


class _TensorRTEngine:
    """Low-level TensorRT engine wrapper with an ORT-compatible ``run()`` API."""

    def __init__(self, engine_path: str | Path):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "tensorrt package is required for the TRT audio backend. "
                "Install with: pip install tensorrt"
            )
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            raise ImportError(
                "cuda-python package is required for the TRT audio backend. "
                "Install with: pip install cuda-python"
            )

        self._trt = trt
        self._cudart = cudart

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        engine_path = Path(engine_path)
        log.info("Loading TRT engine: %s (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()

        err, self._stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        self._inputs: list[str] = []
        self._outputs: list[str] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(name)
            else:
                self._outputs.append(name)

    def _check(self, err, context: str = ""):
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error in {context}: {err}")

    def _alloc_and_copy_h2d(self, host: np.ndarray) -> int:
        cudart = self._cudart
        nbytes = max(host.nbytes, 1)
        err, d_ptr = cudart.cudaMalloc(nbytes)
        self._check(err, "cudaMalloc")
        (err,) = cudart.cudaMemcpy(
            d_ptr, host.ctypes.data, host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        self._check(err, "cudaMemcpy H2D")
        return d_ptr

    def get_inputs(self):
        return [_TensorInfo(n) for n in self._inputs]

    def get_outputs(self):
        return [_TensorInfo(n) for n in self._outputs]

    def run(self, output_names: list[str], input_dict: dict[str, np.ndarray]):
        cudart = self._cudart
        d_ptrs: list[int] = []
        try:
            for name in self._inputs:
                data = np.ascontiguousarray(input_dict[name])
                if data.ndim == 0:
                    data = data.reshape(1)
                self._context.set_input_shape(name, tuple(data.shape))
                d_ptr = self._alloc_and_copy_h2d(data)
                d_ptrs.append(d_ptr)
                self._context.set_tensor_address(name, d_ptr)

            out_meta: dict[str, tuple[np.ndarray, int]] = {}
            for name in self._outputs:
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = _trt_to_np_dtype(self._engine.get_tensor_dtype(name))
                host_buf = np.empty(shape, dtype=dtype)
                nbytes = max(host_buf.nbytes, 1)
                err, d_ptr = cudart.cudaMalloc(nbytes)
                self._check(err, f"cudaMalloc output {name}")
                d_ptrs.append(d_ptr)
                self._context.set_tensor_address(name, d_ptr)
                out_meta[name] = (host_buf, d_ptr)

            if not self._context.execute_async_v3(self._stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            (err,) = cudart.cudaStreamSynchronize(self._stream)
            self._check(err, "cudaStreamSynchronize")

            results: list[np.ndarray] = []
            for name in output_names:
                host_buf, d_ptr = out_meta[name]
                (err,) = cudart.cudaMemcpy(
                    host_buf.ctypes.data, d_ptr, host_buf.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                self._check(err, f"cudaMemcpy D2H {name}")
                results.append(host_buf)
            return results
        finally:
            for p in d_ptrs:
                cudart.cudaFree(p)

    def close(self):
        if hasattr(self, "_stream") and self._stream is not None:
            self._cudart.cudaStreamDestroy(self._stream)
            self._stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TrtAudioTokenizer:
    """Encode waveforms → audio codes and decode codes → waveforms via native TensorRT.

    Args:
        encoder_path: path to encoder ``.engine`` file
        decoder_path: path to decoder ``.engine`` file
        n_quantizers: number of RVQ codebooks (default 32)
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        n_quantizers: int = N_QUANTIZERS,
    ):
        self.n_quantizers = n_quantizers

        encoder_path = Path(encoder_path)
        decoder_path = Path(decoder_path)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder engine not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder engine not found: {decoder_path}")

        self._encoder = _TensorRTEngine(encoder_path)
        self._decoder = _TensorRTEngine(decoder_path)

        self._enc_in = [i.name for i in self._encoder.get_inputs()]
        self._enc_out = [o.name for o in self._encoder.get_outputs()]
        self._dec_in = [i.name for i in self._decoder.get_inputs()]
        self._dec_out = [o.name for o in self._decoder.get_outputs()]

        log.info("TrtAudioTokenizer ready: n_quantizers=%d", n_quantizers)

    def encode(self, waveform: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Encode a waveform to audio codes.

        Returns:
            audio_codes: int64 ``(T', n_vq)``
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
        audio_codes = outputs[0]
        code_lengths = outputs[1]
        code_len = int(code_lengths[0])
        return audio_codes[:, 0, :code_len].T.astype(np.int64)

    def decode(self, audio_codes: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Decode audio codes to a waveform.

        Returns:
            waveform: float32 ``(T,)``
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
        audio = outputs[0]
        audio_lengths = outputs[1]
        length = int(audio_lengths[0])
        return audio[0, 0, :length].astype(np.float32)

    def close(self):
        if hasattr(self, "_encoder"):
            self._encoder.close()
        if hasattr(self, "_decoder"):
            self._decoder.close()
