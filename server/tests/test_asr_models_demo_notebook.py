from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "asr_models_demo.ipynb"


def _notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_asr_demo_has_one_cuda_cell_per_requested_model() -> None:
    notebook = _notebook()
    model_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "asr-model-cell" in cell.metadata.get("tags", [])
    ]
    assert [cell.metadata["tags"][-1] for cell in model_cells] == [
        "model-1",
        "model-2",
        "model-3",
        "model-4",
        "model-5",
        "model-6",
    ]

    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for expected in (
        "Normal Whisper-small",
        "Multiconditioned Whisper-small",
        "Fusion model",
        "Whisper-medium",
        "Normal FastConformer",
        "Multiconditioned FastConformer",
    ):
        assert expected in all_code
    assert "torch.cuda.is_available()" in all_code
    assert "torch.cuda.empty_cache()" in all_code


def test_asr_demo_samples_each_configured_test_tsv_once_for_all_models() -> None:
    notebook = _notebook()
    sampling_cell = next(cell for cell in notebook.cells if cell.id == "sample-test-rows")
    helper_cell = next(cell for cell in notebook.cells if cell.id == "helpers")
    model_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "asr-model-cell" in cell.metadata.get("tags", [])
    ]

    assert "load_split_examples([dataset_dir], 'test')" in helper_cell.source
    assert "SAMPLES_PER_DATASET" in sampling_cell.source
    assert "selected_examples" in sampling_cell.source
    assert all("sample_test_rows" not in cell.source for cell in model_cells)
    assert all(
        "selected_examples" in helper_cell.source or "run_" in cell.source
        for cell in model_cells
    )
