
import pytest
import yaml

from crosssense.config import AppConfig, load_config, save_config


def test_config_roundtrip(tmp_path):
    path = tmp_path / 'config.yaml'
    save_config(AppConfig(), path)
    assert load_config(path) == AppConfig()


@pytest.mark.parametrize('section,key,value', [
    ('data', 'patch_size', 0), ('data', 'patch_size', 4),
    ('data', 'max_cloud_fraction', 1.1), ('data', 'min_valid_fraction', -0.1),
    ('data', 'scene_manifest', 'definitely-missing.csv'),
    ('data', 'target_crs', 'EPSG:4326'), ('data', 'optical_scale', 0),
    ('data', 'split_fractions', [0.5, 0.2, 0.1]),
    ('data', 'sar_bands', [1, 1]), ('model', 'optical_channels', 3),
    ('model', 'freeze_backbones', 'yes'), ('training', 'batch_size', 1),
    ('training', 'batch_size', True), ('training', 'lr', float('nan')),
    ('training', 'device', 'banana'), ('logging', 'level', 'NOPE'),
])
def test_invalid_config(tmp_path, section, key, value):
    payload = AppConfig().to_dict()
    payload[section][key] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=key.split('_')[0] if key not in ('optical_channels',) else 'channel'):
        load_config(path)


def test_missing_and_malformed(tmp_path):
    with pytest.raises(FileNotFoundError, match='Configuration missing'):
        load_config(tmp_path / 'missing.yaml')
    path = tmp_path / 'bad.yaml'
    for text in ('[invalid', '[]', 'unknown: 1', 'data: []', 'data:\n  surprise: 1'):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)
