"""Acceptance scenarios; each assertion observes actual behavior."""
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

from crosssense.cli import main
from crosssense.config import AppConfig
from crosssense.data.pairing import prepare_pairs
from crosssense.data.synthetic import create_synthetic_fixture
from crosssense.evaluation.splits import validate_splits
from crosssense.logging_config import configure_logging
from crosssense.models.training import build_model, load_checkpoint, train_model
from crosssense.retrieval.index import CosineIndex


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def scenes(tmp_path, config):
    return create_synthetic_fixture(tmp_path / 'synthetic', config.data)


def test_scenario_01_happy_path_cpu(tmp_path, capsys):
    output = tmp_path / 'happy'
    assert main(['--log-level', 'ERROR', 'smoke', '--output', str(output), '--epochs', '2']) == 0
    report = json.loads((output / 'evaluation/metrics.json').read_text())
    assert report['data_label'] == 'SYNTHETIC PIPELINE TEST'
    assert report['gallery_pairs'] == 8
    assert set(report['methods']['dual_encoder']) == {'optical_to_sar', 'sar_to_optical'}
    assert (output / 'evaluation/report.html').is_file()
    capsys.readouterr()
    frame = pd.read_csv(output / 'processed/pairs.csv')
    query_id = frame.loc[frame.split == 'test', 'pair_id'].iloc[0]
    assert main(['--log-level', 'ERROR', 'retrieve', '--manifest', str(output / 'processed/pairs.csv'), '--checkpoint', str(output / 'train/best.pt'), '--query-id', query_id, '--k', '3']) == 0
    hits = json.loads(capsys.readouterr().out)
    assert len(hits) == 3 and all({'id', 'score', 'metadata'} <= set(hit) for hit in hits)


def test_scenario_02_no_data(tmp_path, capsys):
    assert main(['build-pairs', '--scenes', str(tmp_path / 'missing.csv')]) == 2
    assert 'missing' in capsys.readouterr().err.lower()


def test_scenario_03_bad_configuration(tmp_path):
    from crosssense.config import load_config
    for payload in ('data:\n  patch_size: -1', 'training:\n  batch_size: 0', 'data:\n  min_overlap: 2', 'data:\n  scene_manifest: nonexistent.csv'):
        path = tmp_path / 'bad.yaml'
        path.write_text(payload)
        with pytest.raises(ValueError):
            load_config(path)


def test_scenario_04_cloudy_optical(scenes, tmp_path, config):
    frame = pd.read_csv(scenes, keep_default_na=False)
    for mask in frame[frame.sensor == 'optical'].cloud_mask_path:
        with rasterio.open(scenes.parent / mask, 'r+') as raster:
            raster.write(np.ones((1, raster.height, raster.width), dtype='float32'))
    manifest = prepare_pairs(scenes, tmp_path / 'cloudy', config.data)
    assert pd.read_csv(manifest).empty
    assert 'cloud_fraction' in set(pd.read_csv(manifest.parent / 'rejections.csv').reason)


def test_scenario_05_time_mismatch(scenes, tmp_path, config):
    frame = pd.read_csv(scenes)
    frame.loc[frame.sensor == 'optical', 'timestamp'] = '2024-03-01T00:00:00Z'
    frame.to_csv(scenes, index=False)
    manifest = prepare_pairs(scenes, tmp_path / 'time', config.data)
    assert pd.read_csv(manifest).empty
    assert 'time_gap' in set(pd.read_csv(manifest.parent / 'rejections.csv').reason)


def test_scenario_06_geographic_leakage(scenes, tmp_path, config):
    manifest = prepare_pairs(scenes, tmp_path / 'paired', config.data)
    frame = pd.read_csv(manifest)
    row = frame.iloc[0].copy()
    row['pair_id'] = 'overlapping-copy'
    row['block_id'] = 'falsified-block'
    row['split'] = 'test' if row['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='Geographic leakage'):
        validate_splits(pd.concat([frame, row.to_frame().T], ignore_index=True))


def test_scenario_07_corrupt_raster_and_missing_band(scenes, tmp_path, config):
    from crosssense.data.preprocess import inspect_raster
    path = tmp_path / 'corrupt.tif'
    path.write_text('not a raster')
    log = configure_logging(tmp_path / 'logs', 'DEBUG', 'corrupt')
    with pytest.raises(ValueError, match='Cannot inspect raster'):
        inspect_raster(path)
    assert 'Traceback' in log.read_text()
    frame = pd.read_csv(scenes)
    with pytest.raises(ValueError, match='bands'):
        inspect_raster(scenes.parent / frame.iloc[0].path, [999])


def test_scenario_08_cpu_only(config):
    from crosssense.models.training import resolve_device
    model = build_model(config).to(resolve_device('cpu'))
    sar, optical = model(torch.zeros(2, 2, 32, 32), torch.zeros(2, 4, 32, 32))
    assert sar.device.type == optical.device.type == 'cpu'
    assert sar.shape == optical.shape == (2, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable; GPU execution not observed')
def test_scenario_09_gpu(config):
    from crosssense.models.training import resolve_device
    model = build_model(config).to(resolve_device('cuda'))
    with torch.no_grad():
        sar, optical = model(torch.zeros(2, 2, 32, 32, device='cuda'), torch.zeros(2, 4, 32, 32, device='cuda'))
    assert sar.is_cuda and optical.is_cuda


def test_scenario_10_interrupted_training(scenes, tmp_path, config):
    manifest = prepare_pairs(scenes, tmp_path / 'paired', config.data)
    config.training.epochs = 1
    train_model(config, manifest, tmp_path / 'train')
    config.training.epochs = 2
    train_model(config, manifest, tmp_path / 'train', tmp_path / 'train/last.pt')
    payload = load_checkpoint(tmp_path / 'train/last.pt', build_model(config))
    assert payload['epoch'] == 2 and len(payload['history']) == 2


def test_scenario_11_empty_gallery():
    with pytest.raises(ValueError, match='gallery is empty'):
        CosineIndex(np.zeros((0, 32)), [], [])


def test_scenario_12_unexpected_exception(tmp_path, monkeypatch, capsys):
    import crosssense.cli as cli
    from crosssense.config import save_config
    config = AppConfig()
    config.logging.output_dir = str(tmp_path / 'logs')
    path = tmp_path / 'config.yaml'
    save_config(config, path)
    def fail(*args):
        raise RuntimeError('injected diagnostic failure')
    monkeypatch.setattr(cli, 'dispatch', fail)
    assert cli.main(['--config', str(path), '--run-id', 'unexpected', 'synthetic']) == 1
    error = capsys.readouterr().err
    assert 'Inspect the traceback' in error
    assert 'Traceback' in (tmp_path / 'logs/unexpected.log').read_text()


def test_cli_help_and_retrieval(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert 'build-pairs' in capsys.readouterr().out



def test_cli_download_stop_returns_failure_status(monkeypatch, capsys):
    from crosssense import cli
    monkeypatch.setattr(cli, 'dispatch', lambda *args: {'status': 'stopped', 'error': 'Size limit exceeded'})
    assert cli.main(['--log-level', 'ERROR', 'download-sen12ms']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'stopped'
