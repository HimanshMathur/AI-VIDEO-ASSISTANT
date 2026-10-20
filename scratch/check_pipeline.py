import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
code = open('app.py', encoding='utf-8').read()

# PIPELINE_STAGES block
block_start = code.find('PIPELINE_STAGES = [')
block_end = code.find('\n]\n', block_start) + 2
stages_block = code[block_start:block_end]
stage_keys = re.findall(r'"(\w+)"', stages_block)
print('PIPELINE_STAGES keys:', stage_keys)

# DEFAULT_STATE pipeline_steps dict
ds_start = code.find('DEFAULT_STATE = {')
ds_end = code.find('\"last_source_id\"', ds_start)
ds_block = code[ds_start:ds_end]
ps_keys = re.findall(r'"(\w+)":\s*"pending"', ds_block)
print('DEFAULT_STATE pipeline_steps keys:', ps_keys)

stages_set = set(stage_keys)
ps_set = set(ps_keys)
print('Missing from DEFAULT_STATE:', stages_set - ps_set)
print('Extra in DEFAULT_STATE (not in STAGES):', ps_set - stages_set)
