from pathlib import Path
import subprocess

root = Path(r'C:\AgapAI\AgapAI')
path = root / 'agapai/api/agapai_api/geography.py'
current = path.read_text(encoding='utf-8-sig')
original = subprocess.check_output(['git', '-c', 'safe.directory=C:/AgapAI/AgapAI', 'show', 'HEAD:agapai/api/agapai_api/geography.py'], cwd=root).decode('utf-8-sig')
start = original.index('def extract_location_name(')
end = original.index('\ndef ', start + 1)
current_start = current.index('def extract_location_name(')
current_end = current.index('\ndef ', current_start + 1)
current = current[:current_start] + original[start:end] + current[current_end:]
current = current.replace('from agapai_api.ner_location_matching import location_mentions\n', '')
path.write_text(current, encoding='utf-8')
