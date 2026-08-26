# Tiny CPU smoke run for global-SGDM update diagnostics.

exec(open('config/smoke_optimizer.py').read())

out_dir = 'out-global-sgdm-diagnostics-smoke'
optimizer_name = 'global_sgdm'
diagnostics_enabled = True
diagnostic_steps = '0'
diagnostic_spectral_matrix_names = 'transformer.h.0.attn.c_proj.weight'
