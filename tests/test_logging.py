import json
import logging

from crosssense.logging_config import configure_logging


def test_log_context_trace_and_redaction(tmp_path):
    path = configure_logging(tmp_path, 'DEBUG', 'test-run')
    logger = logging.getLogger('crosssense.test')
    logger.debug('download https://user:secret@example.org/file?access_token=private')
    try:
        raise RuntimeError('diagnostic stack')
    except RuntimeError:
        logger.exception('Unexpected failure')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]['run_id'] == 'test-run'
    assert records[-1]['module'] == 'crosssense.test'
    assert 'Traceback' in records[-1]['exception']
    text = path.read_text()
    assert 'diagnostic stack' in text
    assert 'secret' not in text and 'private' not in text
    assert '[REDACTED]' in text
