"""ONNX export for MOSS Audio Tokenizer encode/decode paths.

Exports the non-streaming ``_encode_frame`` and ``_decode_frame`` paths as two
separate ONNX models using ``torch.onnx.export(dynamo=True)``.  The residual
quantiser loop is unrolled over all ``num_quantizers`` steps; inactive steps
are masked via ``torch.where`` so ``n_quantizers`` remains a runtime input.
All sequence-length axes are declared dynamic via ``torch.export.Dim``.

Usage
-----
>>> python -m moss_audio_tokenizer.export_onnx \
...     --model_path OpenMOSS-Team/MOSS-Audio-Tokenizer \
...     --output_dir ./onnx_export
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from ..modeling_moss_audio_tokenizer import (
    MossAudioTokenizerModel,
    StreamingModule,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-export preparation
# ---------------------------------------------------------------------------

def remove_weight_norm(model: nn.Module) -> None:
    """Remove ``weight_norm`` / ``parametrize`` wrappers from all Conv1d layers."""
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            try:
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight")
            except ValueError:
                pass


def prepare_model_for_export(model: MossAudioTokenizerModel) -> MossAudioTokenizerModel:
    """Eval mode, strip weight-norm, assert non-streaming."""
    model.eval()
    remove_weight_norm(model)
    for name, module in model.named_modules():
        if isinstance(module, StreamingModule) and module._streaming_state is not None:
            raise RuntimeError(
                f"Module '{name}' is in streaming mode. "
                "Call model._stop_streaming() before export."
            )
    return model


# ---------------------------------------------------------------------------
# Exportable encoder
# ---------------------------------------------------------------------------

class ExportableEncoder(nn.Module):
    """Wraps the encoder + residual-quantiser path for ``torch.export``.

    The RVQ loop is unrolled over all ``num_quantizers`` steps.  Inactive
    steps (where ``i >= n_quantizers``) are masked with ``torch.where`` so
    they contribute zero to both the residual update and the output indices.
    This avoids ``torch.cond`` (whose branch-function closures cannot
    reference ``nn.Module`` attributes under TorchDynamo).

    .. important::

       ``T`` **must** be a multiple of ``downsample_rate`` (1920).  Pad the
       waveform *before* calling this model — dynamic padding cannot be
       reliably captured by ``torch.export``/ONNX because the pad length
       depends on a symbolic dimension.

    Inputs
        input_values : ``(B, 1, T)`` float — raw waveform,
                       ``T`` divisible by ``downsample_rate``
        n_quantizers : ``()`` long scalar — how many RVQ layers to use

    Outputs
        audio_codes       : ``(num_quantizers, B, T')`` long
        audio_codes_lengths : ``(B,)`` long
    """

    def __init__(self, model: MossAudioTokenizerModel) -> None:
        super().__init__()
        self.encoder_modules = model.encoder
        self.downsample_rate: int = model.downsample_rate

        quantizer = model.quantizer
        self.quantizer_input_proj = quantizer.input_proj
        self.quantizers = quantizer.quantizers
        self.num_quantizers: int = quantizer.num_quantizers
        self.rvq_dim: int = quantizer.rvq_dim

    def forward(
        self,
        input_values: torch.Tensor,
        n_quantizers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, T = input_values.shape

        input_lengths = torch.full(
            (B,), T, device=input_values.device, dtype=torch.long,
        )

        e: torch.Tensor = input_values
        e_lengths: torch.Tensor = input_lengths
        for enc_module in self.encoder_modules:
            e, e_lengths = enc_module(e, e_lengths)

        z = self.quantizer_input_proj(e).float()
        _, _, max_time = z.shape

        mask = (
            torch.arange(max_time, device=z.device).unsqueeze(0).expand(B, -1)
            < e_lengths.unsqueeze(1)
        )
        mask_expanded = mask.unsqueeze(1).to(z.dtype)

        residual = z.clone()
        indices_list: list[torch.Tensor] = []
        zero_idx = torch.zeros(B, max_time, device=z.device, dtype=torch.long)

        for i in range(self.num_quantizers):
            active = n_quantizers > i                          # bool scalar
            active_float = active.to(z.dtype)                  # 0.0 or 1.0

            masked_residual = residual * mask_expanded
            z_q_i, indices_i, _ = self.quantizers[i](masked_residual)

            residual = residual - z_q_i * mask_expanded * active_float
            indices_list.append(torch.where(active, indices_i, zero_idx))

        all_indices = torch.stack(indices_list, dim=0)         # (num_q, B, T')
        return all_indices, e_lengths


# ---------------------------------------------------------------------------
# Exportable decoder
# ---------------------------------------------------------------------------

class ExportableDecoder(nn.Module):
    """Wraps the de-quantiser + decoder path for ``torch.export``.

    Like the encoder, the RVQ decode loop is unrolled and uses
    ``torch.where`` masking instead of ``torch.cond``.

    Inputs
        audio_codes  : ``(num_quantizers, B, T')`` long — always padded to
                       ``num_quantizers`` (32) along dim-0; unused slots
                       are ignored via ``n_quantizers``.
        n_quantizers : ``()`` long scalar — how many RVQ layers to sum

    Outputs
        audio         : ``(B, 1, T)`` float
        audio_lengths : ``(B,)`` long
    """

    def __init__(self, model: MossAudioTokenizerModel) -> None:
        super().__init__()
        self.decoder_modules = model.decoder

        quantizer = model.quantizer
        self.quantizer_output_proj = quantizer.output_proj
        self.quantizers = quantizer.quantizers
        self.num_quantizers: int = quantizer.num_quantizers
        self.rvq_dim: int = quantizer.rvq_dim

    def forward(
        self,
        audio_codes: torch.Tensor,
        n_quantizers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, B, T = audio_codes.shape

        # Collect all quantiser embeddings, then stack+sum in one shot.
        # This avoids 32 chained Add ops in the ONNX graph which cause
        # ORT's symbolic shape inference to lose track of the rank
        # (resulting in a bogus 5-D shape and an AssertionError during
        # quantization preprocessing).
        parts: list[torch.Tensor] = []
        for i in range(self.num_quantizers):
            codes_i = audio_codes[i]
            active_float = (n_quantizers > i).to(torch.float32)
            quantized_i = self.quantizers[i].decode_code(codes_i).float()
            parts.append(quantized_i * active_float)

        emb = torch.stack(parts, dim=0).sum(dim=0)           # (B, rvq_dim, T)
        emb = self.quantizer_output_proj(emb)

        codes_lengths = torch.full((B,), T, device=audio_codes.device, dtype=torch.long)
        d: torch.Tensor = emb
        d_lengths: torch.Tensor = codes_lengths
        for dec_module in self.decoder_modules:
            d, d_lengths = dec_module(d, d_lengths)

        return d, d_lengths


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_encoder(
    model: MossAudioTokenizerModel,
    output_path: str | Path = "encoder.onnx",
    *,
    max_batch: int = 32,
    max_seconds: float = 300.0,
) -> None:
    """Export the encoder to ONNX with dynamic batch and sequence length.

    The caller must pre-pad the waveform so that ``T`` is a multiple of
    ``downsample_rate`` (1920) before feeding it to the exported model.
    """
    from torch.export import Dim

    model = prepare_model_for_export(model)
    encoder = ExportableEncoder(model).eval()

    downsample_rate = model.downsample_rate
    sr = model.sampling_rate
    max_samples = int(max_seconds * sr)
    max_samples = (
        max_samples
        - (max_samples % downsample_rate)
        + downsample_rate
    )

    batch = Dim("batch", min=1, max=max_batch)
    # T must be a multiple of downsample_rate at runtime.
    seq_len = Dim("seq_len", min=downsample_rate, max=max_samples)

    dummy_wav = torch.randn(1, 1, downsample_rate * 10)
    dummy_nq = torch.tensor(32, dtype=torch.long)

    logger.info("Exporting encoder to %s …", output_path)
    onnx_program = torch.onnx.export(
        encoder,
        (dummy_wav, dummy_nq),
        dynamo=True,
        dynamic_shapes={
            "input_values": {0: batch, 2: seq_len},
            "n_quantizers": {},
        },
    )
    onnx_program.save(str(output_path))
    logger.info("Encoder saved to %s", output_path)


def export_decoder(
    model: MossAudioTokenizerModel,
    output_path: str | Path = "decoder.onnx",
    *,
    max_batch: int = 32,
    max_seconds: float = 300.0,
) -> None:
    """Export the decoder to ONNX with dynamic batch and code length."""
    from torch.export import Dim

    model = prepare_model_for_export(model)
    decoder = ExportableDecoder(model).eval()

    downsample_rate = model.downsample_rate
    sr = model.sampling_rate
    max_codes = int(max_seconds * sr / downsample_rate)

    batch = Dim("batch", min=1, max=max_batch)
    code_len = Dim("code_len", min=1, max=max_codes)

    num_q = model.quantizer.num_quantizers
    dummy_codes = torch.randint(0, 1024, (num_q, 1, 10))
    dummy_nq = torch.tensor(num_q, dtype=torch.long)

    logger.info("Exporting decoder to %s …", output_path)
    onnx_program = torch.onnx.export(
        decoder,
        (dummy_codes, dummy_nq),
        dynamo=True,
        dynamic_shapes={
            "audio_codes": {1: batch, 2: code_len},
            "n_quantizers": {},
        },
    )
    onnx_program.save(str(output_path))
    logger.info("Decoder saved to %s", output_path)


# ---------------------------------------------------------------------------
# ORT symbolic-shape-inference bug fix
# ---------------------------------------------------------------------------

def patch_ort_shape_inference() -> None:
    """Patch ORT ``SymbolicShapeInference`` to handle patterns emitted by
    ``torch.onnx.export(dynamo=True)`` that the stock implementation cannot
    resolve.

    **Bug 1 – Shape start/end** (opset >= 15): ``_infer_Shape`` ignores the
    ``start``/``end`` attributes, storing the full input shape.  Any downstream
    ``Range`` whose limit derives from a sliced ``Shape`` receives a
    multi-element list and ``as_scalar`` crashes.

    **Bug 2 – Missing ReduceL2 handler**: No ``_infer_ReduceL2`` exists, so
    the output shape of ReduceL2 (used by RVQ L2-normalisation in the encoder)
    is left unresolved.  Downstream ``Expand`` then receives ``None`` for the
    input shape and crashes in ``_broadcast_shapes``.
    """
    from onnxruntime.tools.symbolic_shape_infer import (
        SymbolicShapeInference,
        get_attribute,
        get_opset,
        handle_negative_axis,
    )
    from onnx import helper

    # --- Fix 1: Shape with start/end attributes ---
    def _patched_infer_Shape(self, node):  # noqa: N802
        sympy_shape = self._get_sympy_shape(node, 0)
        start = get_attribute(node, "start", 0)
        end = get_attribute(node, "end", len(sympy_shape))
        if end < 0:
            end = len(sympy_shape) + end
        self.sympy_data_[node.output[0]] = sympy_shape[start:end]

    SymbolicShapeInference._infer_Shape = _patched_infer_Shape

    # --- Fix 2: ReduceL2 handler (mirrors ReduceSum for opset >= 18) ---
    def _patched_infer_ReduceL2(self, node):  # noqa: N802
        keep_dims = get_attribute(node, "keepdims", 1)
        if get_opset(self.out_mp_) >= 18 and len(node.input) > 1:
            axes = self._try_get_value(node, 1)
            vi = self.known_vi_[node.output[0]]
            if axes is None:
                assert keep_dims
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        self._get_shape(node, 0),
                    )
                )
            else:
                shape = self._get_shape(node, 0)
                output_shape = []
                axes = [handle_negative_axis(a, len(shape)) for a in axes]
                for i, d in enumerate(shape):
                    if i in axes:
                        if keep_dims:
                            output_shape.append(1)
                    else:
                        output_shape.append(d)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        output_shape,
                    )
                )

    SymbolicShapeInference._infer_ReduceL2 = _patched_infer_ReduceL2
    SymbolicShapeInference.dispatcher_ = None  # force re-init

    # The dispatcher_ is a class-level dict built in __init__; we need to
    # register our new handler there.  We do this by wrapping __init__.
    _orig_init = SymbolicShapeInference.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.dispatcher_["ReduceL2"] = self._infer_ReduceL2

    SymbolicShapeInference.__init__ = _patched_init

    logger.info(
        "Patched ORT SymbolicShapeInference "
        "(_infer_Shape start/end, _infer_ReduceL2)"
    )


# ---------------------------------------------------------------------------
# INT8 static quantization (requires onnxruntime)
# ---------------------------------------------------------------------------

def quantize_onnx(
    input_onnx: str | Path,
    output_onnx: str | Path,
    calibration_data: list[dict[str, np.ndarray]],
    *,
    exclude_node_names: list[str] | None = None,
) -> None:
    """Apply INT8 static quantization with ONNX Runtime.

    Parameters
    ----------
    input_onnx : path to the FP32 ``.onnx`` file.
    output_onnx : destination path for the quantised model.
    calibration_data : list of feed-dicts (numpy arrays) used for
        activation range calibration.  ~100 representative samples
        is usually sufficient.
    exclude_node_names : ONNX node names to keep in FP32 (e.g. codebook
        embedding nodes).
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, data: list[dict[str, np.ndarray]]):
            self._data = iter(data)

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._data, None)

    quantize_static(
        str(input_onnx),
        str(output_onnx),
        calibration_data_reader=_Reader(calibration_data),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=exclude_node_names or [],
    )
    logger.info("Quantised model saved to %s", output_onnx)


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_onnx(
    model: MossAudioTokenizerModel,
    encoder_onnx: str | Path,
    decoder_onnx: str | Path,
    *,
    duration_seconds: float = 1.0,
    n_quantizers: int = 32,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare ONNX Runtime outputs against the PyTorch model.

    Returns a dict with ``encoder_max_abs_err`` and ``decoder_max_abs_err``.
    """
    import numpy as np
    import onnxruntime as ort

    sr = model.sampling_rate
    T = int(duration_seconds * sr)
    T = T - (T % model.downsample_rate)
    dummy_wav = torch.randn(1, 1, T)
    nq_tensor = torch.tensor(n_quantizers, dtype=torch.long)

    # --- PyTorch reference ---
    model.eval()
    enc_ref = model._encode_frame(dummy_wav, n_quantizers=n_quantizers)
    dec_ref = model._decode_frame(enc_ref.audio_codes, enc_ref.audio_codes_lengths)

    # --- ONNX Runtime encoder ---
    enc_sess = ort.InferenceSession(str(encoder_onnx))
    enc_inputs = {
        "input_values": dummy_wav.numpy(),
        "n_quantizers": nq_tensor.numpy(),
    }
    enc_codes_ort, enc_lengths_ort = enc_sess.run(None, enc_inputs)

    enc_err = float(np.max(np.abs(enc_ref.audio_codes.numpy() - enc_codes_ort)))

    # --- ONNX Runtime decoder ---
    dec_sess = ort.InferenceSession(str(decoder_onnx))
    dec_inputs = {
        "audio_codes": enc_codes_ort,
        "n_quantizers": nq_tensor.numpy(),
    }
    dec_audio_ort, dec_lengths_ort = dec_sess.run(None, dec_inputs)

    dec_err = float(np.max(np.abs(dec_ref.audio.numpy() - dec_audio_ort)))

    results = {
        "encoder_max_abs_err": enc_err,
        "decoder_max_abs_err": dec_err,
    }
    for key, val in results.items():
        status = "PASS" if val <= atol else "FAIL"
        logger.info("  %s = %.6e  [%s, atol=%.0e]", key, val, status, atol)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export MOSS Audio Tokenizer to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        help="HuggingFace repo-id or local path to model",
    )
    parser.add_argument("--output_dir", type=str, default="./onnx_export")
    parser.add_argument("--validate", action="store_true", help="Run numerical validation after export")
    parser.add_argument("--quantize", action="store_true", help="Quantize the exported models to INT8")
    parser.add_argument("--max_batch", type=int, default=32)
    parser.add_argument("--max_seconds", type=float, default=300.0)
    args = parser.parse_args()

    from transformers import AutoModel

    logger.info("Loading model from %s …", args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model = cast(MossAudioTokenizerModel, model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder.onnx"

    export_encoder(
        model,
        encoder_path,
        max_batch=args.max_batch,
        max_seconds=args.max_seconds,
    )
    export_decoder(
        model,
        decoder_path,
        max_batch=args.max_batch,
        max_seconds=args.max_seconds,
    )

    if args.quantize:
        patch_ort_shape_inference()
        logger.info("Quantizing models to INT8 …")
        
        # 1. Encoder quantization
        encoder_q_path = encoder_path.with_name(encoder_path.stem + "_int8.onnx")
        encoder_cal = [
            {
                "input_values": torch.randn(1, 1, model.downsample_rate * 10).numpy(),
                "n_quantizers": torch.tensor(32, dtype=torch.long).numpy(),
            }
            for _ in range(10)
        ]
        quantize_onnx(encoder_path, encoder_q_path, encoder_cal)
        
        # 2. Decoder quantization
        decoder_q_path = decoder_path.with_name(decoder_path.stem + "_int8.onnx")
        num_q = model.quantizer.num_quantizers
        decoder_cal = [
            {
                "audio_codes": torch.randint(0, 1024, (num_q, 1, 10)).numpy(),
                "n_quantizers": torch.tensor(num_q, dtype=torch.long).numpy(),
            }
            for _ in range(10)
        ]
        quantize_onnx(decoder_path, decoder_q_path, decoder_cal)

    if args.validate:
        logger.info("Running numerical validation …")
        results = validate_onnx(model, encoder_path, decoder_path)
        print("\nNumerical Validation Results (FP32 vs PyTorch):")
        for key, val in results.items():
            print(f"  {key}: {val:.6e}")

        if args.quantize:
            logger.info("Running numerical validation (INT8) …")
            encoder_q_path = encoder_path.with_name(encoder_path.stem + "_int8.onnx")
            decoder_q_path = decoder_path.with_name(decoder_path.stem + "_int8.onnx")
            results_q = validate_onnx(model, encoder_q_path, decoder_q_path)
            print("\nNumerical Validation Results (INT8 vs PyTorch):")
            for key, val in results_q.items():
                print(f"  {key}: {val:.6e}")


if __name__ == "__main__":
    main()
