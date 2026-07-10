n_files=100

for i in range(n_files):
    with open(f"input{i:03d}.yaml", "w") as f:
        f.write(f"seed: {i}\n")
        f.write(f"games: 10000\n")

with open("tasks.sh", "w") as f:
    for i in range(n_files):
        f.write(f".venv/bin/python main.py input{i:03d}.yaml\n")

