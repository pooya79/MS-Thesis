import csv
import io
import json
import tarfile

import pytest

from ml.speech_data.prepare_cv25_official import CLEAN, DEGRADED, main, prepare


def fixture_data(tmp_path, overlap=False):
    root = tmp_path / 'data'
    for name in (CLEAN, DEGRADED):
        (root / name / 'clips').mkdir(parents=True)
    archive = tmp_path / 'cv.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for split, key in [('train', 'a'), ('dev', 'a' if overlap else 'b'), ('test', 'c')]:
            content = f'path\tsentence\tclient_id\n{key}.mp3\tofficial {key}\tspeaker-{key}\n'.encode()
            member = tarfile.TarInfo(f'cv/fa/{split}.tsv')
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    records = []
    for key in 'abcd':
        (root / CLEAN / 'clips' / f'{key}.flac').write_bytes(b'audio')
        (root / DEGRADED / 'clips' / f'{key}-v0.flac').write_bytes(b'degraded')
        records.append(dict(clean_path=str(root / CLEAN / 'clips' / f'{key}.wav'),
                            degraded_path=str(root / DEGRADED / 'clips' / f'{key}-v0.wav'),
                            degraded_id=key, split='train', sentence='old',
                            degradation={'seed': 42, 'codec': 'amr', 'split': 'train'}))
    (root / DEGRADED / 'degraded_to_clean.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return archive, root


def test_official_membership_reassigns_variants_and_excludes_test(tmp_path):
    archive, root = fixture_data(tmp_path)
    output = tmp_path / 'official'
    report = prepare(archive, root, output)
    assert report['clean_counts'] == dict(train=1, dev=1, test=1)
    assert report['counts']['reassigned_variants'] == 1
    assert report['counts']['excluded_variants'] == 2
    assert report['missing_degraded_sources'] == dict(train=[], dev=[])
    records = [json.loads(s) for s in (output / DEGRADED / 'degraded_to_clean.jsonl').read_text().splitlines()]
    assert [(r['degraded_id'], r['split']) for r in records] == [('a', 'train'), ('b', 'dev')]
    assert records[1]['sentence'] == 'official b'
    assert records[1]['degradation'] == records[1]['original_mapping']['degradation']
    assert records[1]['original_mapping']['split'] == 'train'
    assert (output / CLEAN / 'clips/a.flac').stat().st_ino == (root / CLEAN / 'clips/a.flac').stat().st_ino
    with (output / CLEAN / 'dev.tsv').open() as stream:
        assert list(csv.DictReader(stream, delimiter='\t'))[0]['client_id'] == 'speaker-b'
    with pytest.raises(FileExistsError):
        prepare(archive, root, output)


def test_overlap_fails_before_output(tmp_path):
    archive, root = fixture_data(tmp_path, overlap=True)
    with pytest.raises(ValueError, match='overlapping'):
        prepare(archive, root, tmp_path / 'official')
    assert not (tmp_path / 'official').exists()


def test_missing_selected_audio_fails(tmp_path):
    archive, root = fixture_data(tmp_path)
    (root / DEGRADED / 'clips/b-v0.flac').unlink()
    with pytest.raises(FileNotFoundError):
        prepare(archive, root, tmp_path / 'official')


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert all(flag in help_text for flag in ('--archive', '--source-root', '--output', 'data/cv25-official'))


def test_recovers_missing_clean_audio_from_archive(tmp_path):
    archive, root = fixture_data(tmp_path)
    from ml.speech_data.prepare_cv25_official import official_tables
    tables = official_tables(archive)
    (root / CLEAN / 'clips/a.flac').unlink()
    with tarfile.open(archive, 'w:gz') as tar:
        for split, content in tables.items():
            member = tarfile.TarInfo(f'cv/fa/{split}.tsv')
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        member = tarfile.TarInfo('cv/fa/clips/a.mp3')
        member.size = 3
        tar.addfile(member, io.BytesIO(b'mp3'))
    output = tmp_path / 'official'
    prepare(archive, root, output)
    assert (output / CLEAN / 'clips/a.mp3').read_bytes() == b'mp3'


def test_launcher_help():
    import subprocess
    result = subprocess.run(['bash', 'ml/speech_data/scripts/run_cv25_tiny_official.sh', '--help'], capture_output=True, text=True, check=True)
    assert 'No arguments' in result.stdout
    assert 'stops on first failure' in result.stdout
