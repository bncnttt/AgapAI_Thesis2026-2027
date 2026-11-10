import ast
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_algorithm import get_nlp

tree = ast.parse(Path(r'C:\AgapAI\.dependency-setup\verify_extraction.py').read_text(encoding='utf-8'))
samples = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'samples' for t in node.targets))
nlp = get_nlp()
print('Pipelines:', nlp.pipe_names, flush=True)
print('NER labels:', nlp.get_pipe('ner').labels, flush=True)
for text in samples + ['Sunog sa Banilad', 'May baha sa Barangay Sunog, Balamban']:
    doc = nlp(text)
    print('\nEntities:', [(ent.text, ent.label_) for ent in doc.ents], flush=True)
    print('Token predictions:', [(token.text, token.pos_, token.dep_) for token in doc], flush=True)
