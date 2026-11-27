import json
import subprocess

from tools import external_snapshot as x


def _repo(path):
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    (path / 'a.json').write_text('[1]')
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x'], check=True)


def test_observe_records_head_and_hashes_without_pull(tmp_path):
    repo = tmp_path / 'src'; repo.mkdir(); _repo(repo)
    rec = x.observe('src', repo, ['a.json', 'missing.json'], pull=False)
    assert rec['changed'] is False and len(rec['head']) == 40
    assert rec['files']['a.json'] and rec['files']['missing.json'] is None


def test_main_appends_error_record_and_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(x, 'SOURCES', {'absent': ['a.json']})
    assert x.main(['--external', str(tmp_path), '--no-pull']) == 1
    line = json.loads((tmp_path / 'observations.jsonl').read_text().splitlines()[0])
    assert line['source'] == 'absent' and 'error' in line
