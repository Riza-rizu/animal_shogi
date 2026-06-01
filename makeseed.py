n_files=19

for i in range(n_files):
    with open(f"input{i:03d}.yaml", "w") as f:
        f.write(f"seed: {i}\n")
        f.write(f"games: 10\n")

with open("tasks.sh", "w") as f:
    for i in range(n_files):
        f.write(f"python main.py input{i:03d}.yaml\n")

