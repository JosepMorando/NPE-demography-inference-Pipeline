import os

target_file = "python/validate_ppc.py"

if not os.path.exists(target_file):
    print(f"Error: {target_file} not found!")
    exit(1)

with open(target_file, "r") as f:
    lines = f.readlines()

new_lines = []
patched = False

for line in lines:
    # We look for the call to render_slim_script or run_slim to inject our check beforehand
    if ("render_slim_script" in line or "run_slim(" in line) and not patched and "def " not in line:
        indent = line[:len(line) - len(line.lstrip())]
        
        # Inject the fix
        new_lines.append(f"{indent}# --- CRITICAL FIX: Dynamic GENS extension ---\n")
        new_lines.append(f"{indent}# Ensure simulation is long enough for the deepest split time in this sample\n")
        new_lines.append(f"{indent}max_t_sample = max([float(v) for k,v in params.items() if k.startswith('T_')] + [0])\n")
        new_lines.append(f"{indent}burnin_val = float(cfg['simulation'].get('burnin', 0))\n")
        new_lines.append(f"{indent}# Use config GENS or (max_t + burnin + buffer), whichever is larger\n")
        new_lines.append(f"{indent}config_gens = float(cfg['simulation'].get('gens', 0))\n")
        new_lines.append(f"{indent}needed_gens = max_t_sample + burnin_val + 500\n")
        new_lines.append(f"{indent}params['GENS'] = max(config_gens, needed_gens)\n")
        new_lines.append(f"{indent}# ------------------------------------------\n")
        
        patched = True
    
    new_lines.append(line)

with open(target_file, "w") as f:
    f.writelines(new_lines)

if patched:
    print(f"Successfully patched {target_file}")
else:
    print("WARNING: Could not find insertion point. File might already be modified or structure differs.")