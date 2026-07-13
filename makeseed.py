n_files = 100
games_per_file = 10000

for i in range(n_files):
    with open(f"input{i:03d}.yaml", "w") as f:
        f.write(f"seed: {i}\n")
        f.write(f"games: {games_per_file}\n")
        f.write("p1: ai\n")
        f.write("p2: ai\n")
        f.write("model_path: animal_shogi_model.keras\n")
        f.write("ratio: 4.0\n")

with open("tasks.sh", "w") as f:
    for i in range(n_files):
        f.write(f".venv/bin/python main.py input{i:03d}.yaml\n")

