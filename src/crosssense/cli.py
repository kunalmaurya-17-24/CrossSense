"""CrossSense command-line experiments. Run from the project root."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Sequence

from crosssense.config import AppConfig, load_config, save_config
from crosssense.logging_config import configure_logging

LOG = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    """Create the public command interface."""
    cli = argparse.ArgumentParser(prog='crosssense', description='Cross-modal optical/SAR patch retrieval research pipeline')
    cli.add_argument('--config', default='configs/default.yaml', help='Validated YAML config')
    cli.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    cli.add_argument('--run-id', default=None)
    commands = cli.add_subparsers(dest='command', required=True)
    synthetic = commands.add_parser('synthetic', help='Create explicitly synthetic georeferenced test scenes')
    synthetic.add_argument('--output', default='data/synthetic')
    prepare = commands.add_parser('build-pairs', help='Align local scenes, extract patches and assign geographic splits')
    prepare.add_argument('--scenes', required=True)
    prepare.add_argument('--output', default='data/processed')
    sen = commands.add_parser('download-sen12ms', help='One bounded official-source attempt; oversized archives are refused')
    sen.add_argument('--output', default='outputs/data_access/sen12ms')
    sen.add_argument('--max-mib', type=float, default=50)
    sen.add_argument('--time-limit', type=float, default=90)
    catalog = commands.add_parser('catalog-india', help='Fetch a bounded public CDSE metadata manifest for Pune')
    catalog.add_argument('--output', default='data/manifests/india_catalogue.json')
    imp = commands.add_parser('import-sen12ms', help='Index local official SEN12MS matched filenames without fabricating times')
    imp.add_argument('--root', required=True)
    imp.add_argument('--output', default='data/manifests/sen12ms_scenes.csv')
    train = commands.add_parser('train', help='Train two encoders with geographically masked symmetric InfoNCE')
    train.add_argument('--manifest', default='data/processed/pairs.csv')
    train.add_argument('--output', default='outputs/train')
    train.add_argument('--resume')
    train.add_argument('--epochs', type=int)
    evaluate = commands.add_parser('evaluate', help='Compare baseline and checkpoint in both sensor directions')
    evaluate.add_argument('--manifest', default='data/processed/pairs.csv')
    evaluate.add_argument('--checkpoint', default='outputs/train/best.pt')
    evaluate.add_argument('--output', default='outputs/evaluation')
    evaluate.add_argument('--split', default='test', choices=['validation', 'test'])
    retrieve = commands.add_parser('retrieve', help='Retrieve matches for one manifest query ID')
    retrieve.add_argument('--manifest', default='data/processed/pairs.csv')
    retrieve.add_argument('--checkpoint', default='outputs/train/best.pt')
    retrieve.add_argument('--query-id', required=True)
    retrieve.add_argument('--direction', default='optical-to-sar', choices=['optical-to-sar', 'sar-to-optical'])
    retrieve.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    retrieve.add_argument('--k', type=int, default=5)
    check = commands.add_parser('check-splits', help='Check spatial blocks, footprints and pair IDs for leakage')
    check.add_argument('--manifest', default='data/processed/pairs.csv')
    inspect = commands.add_parser('inspect', help='Inspect a local raster and its geospatial metadata')
    inspect.add_argument('path')
    smoke = commands.add_parser('smoke', help='Prepare/train/evaluate a tiny SYNTHETIC CPU experiment')
    smoke.add_argument('--output', default='outputs/smoke')
    smoke.add_argument('--epochs', type=int, default=4)
    return cli


def _retrieve(args: argparse.Namespace, config: AppConfig) -> list[dict[str, object]]:
    from crosssense.data.dataset import PairedPatchDataset, load_manifest
    from crosssense.evaluation.splits import validate_splits
    from crosssense.models.training import (
        build_model,
        embed_dataset,
        load_checkpoint,
        resolve_device,
        validate_manifest_preprocessing,
        validate_preprocessing,
    )
    from crosssense.retrieval.index import CosineIndex
    frame = load_manifest(args.manifest)
    validate_splits(frame, config.data.split_buffer_m)
    band_schema = validate_manifest_preprocessing(config, frame)
    frame = frame[frame.split == args.split].reset_index(drop=True)
    if args.query_id not in set(frame.pair_id):
        raise ValueError(f'Query ID {args.query_id} is absent from split={args.split}; inspect the pair manifest')
    dataset = PairedPatchDataset(frame)
    model = build_model(config)
    payload = load_checkpoint(args.checkpoint, model)
    validate_preprocessing(config, payload['config'])
    if payload.get('band_schema') != band_schema:
        raise ValueError('Checkpoint semantic band order differs from prepared patches; use matching bands')
    device = resolve_device(config.training.device)
    model.to(device)
    sar, optical = embed_dataset(model, dataset, device, config.training.batch_size)
    gallery, queries = (sar, optical) if args.direction == 'optical-to-sar' else (optical, sar)
    index = CosineIndex(gallery, frame.pair_id.tolist(), frame.to_dict('records'))
    query_index = frame.pair_id.tolist().index(args.query_id)
    hits = index.search(queries[query_index], args.k)[0]
    LOG.info('Retrieval query=%s direction=%s gallery_count=%d returned=%d', args.query_id, args.direction, len(frame), len(hits))
    return hits


def dispatch(args: argparse.Namespace, config: AppConfig) -> object:
    """Execute a validated subcommand and return a serializable outcome."""
    if args.command == 'synthetic':
        from crosssense.data.synthetic import create_synthetic_fixture
        return {'source': 'synthetic', 'scenes': str(create_synthetic_fixture(args.output, config.data, config.seed))}
    if args.command == 'build-pairs':
        from crosssense.data.pairing import prepare_pairs
        return {'manifest': str(prepare_pairs(args.scenes, args.output, config.data, config.seed))}
    if args.command == 'import-sen12ms':
        from crosssense.data.pairing import import_sen12ms
        return {'scenes': str(import_sen12ms(args.root, args.output))}
    if args.command == 'download-sen12ms':
        from crosssense.data.download import attempt_sen12ms
        result = attempt_sen12ms(args.output, int(args.max_mib * 1024**2), args.time_limit)
        if result['status'] != 'verified':
            LOG.warning('Data acquisition did not complete: %s', result.get('error', result['status']))
        return result
    if args.command == 'catalog-india':
        from crosssense.data.catalog import query_india_catalogue
        return query_india_catalogue(args.output)
    if args.command == 'train':
        from crosssense.models.training import train_model
        if args.epochs is not None:
            config.training.epochs = args.epochs
            config.validate()
        return {'checkpoint': str(train_model(config, args.manifest, args.output, args.resume))}
    if args.command == 'evaluate':
        from crosssense.evaluation.reports import evaluate_experiment
        return evaluate_experiment(config, args.manifest, args.checkpoint, args.output, args.split)
    if args.command == 'retrieve':
        return _retrieve(args, config)
    if args.command == 'check-splits':
        from crosssense.data.dataset import load_manifest
        from crosssense.evaluation.splits import validate_splits
        frame = load_manifest(args.manifest)
        validate_splits(frame, config.data.split_buffer_m)
        return {'status': 'passed', 'counts': frame.split.value_counts().to_dict()}
    if args.command == 'inspect':
        from dataclasses import asdict

        from crosssense.data.preprocess import inspect_raster
        return asdict(inspect_raster(args.path))
    if args.command == 'smoke':
        from crosssense.data.pairing import prepare_pairs
        from crosssense.data.synthetic import create_synthetic_fixture
        from crosssense.evaluation.reports import evaluate_experiment
        from crosssense.models.training import train_model
        output = Path(args.output)
        config.training.device = 'cpu'
        config.training.epochs = args.epochs
        config.validate()
        save_config(config, output / 'config.yaml')
        scenes = create_synthetic_fixture(output / 'synthetic', config.data, config.seed)
        manifest = prepare_pairs(scenes, output / 'processed', config.data, config.seed)
        checkpoint = train_model(config, manifest, output / 'train')
        return evaluate_experiment(config, manifest, checkpoint, output / 'evaluation')
    raise ValueError(f'Unknown command {args.command}')


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI with actionable errors and full unexpected exception logging."""
    args = parser().parse_args(argv)
    run_id = args.run_id or f'{args.command}-{uuid.uuid4().hex[:8]}'
    # Bootstrap logs also capture invalid configuration errors.
    log_path = configure_logging('outputs/logs', args.log_level or 'INFO', run_id)
    try:
        config = load_config(args.config)
        level = args.log_level or config.logging.level
        if config.logging.output_dir != 'outputs/logs' or level != 'INFO':
            log_path = configure_logging(config.logging.output_dir, level, run_id)
        LOG.info('Run started command=%s seed=%d config=%s', args.command, config.seed, args.config)
        result = dispatch(args, config)
        print(json.dumps(result, indent=2, default=str))
        if args.command == 'download-sen12ms' and result.get('status') != 'verified':
            return 2
        if args.command == 'catalog-india' and any(q.get('status') != 'ok' for q in result.get('queries', [])):
            return 2
        LOG.info('Run completed command=%s', args.command)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        LOG.error('%s', exc)
        print(f'CrossSense: {exc}\nSee {log_path} and docs/data_sources.md for setup.', file=sys.stderr)
        return 2
    except Exception:
        LOG.exception('Unexpected failure in command=%s', args.command)
        print(f'CrossSense: unexpected failure during {args.command}. Inspect the traceback in {log_path}.', file=sys.stderr)
        return 1
