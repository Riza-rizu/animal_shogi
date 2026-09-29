n_files = 15  # サーバーの 15 ワーカー (mpirun -np 16: マスター1 + ワーカー15) と完全一致
games_per_file = 1000  # 1ファイルあたり1,000試合 (合計15,000試合、約15分で完了)

for i in range(n_files):
    with open(f"input{i:03d}.yaml", "w") as f:
        f.write(f"seed: {i}\n")
        f.write(f"games: {games_per_file}\n")
        f.write("p1: ai\n")
        f.write("p2: ai\n")
        f.write("model_path: animal_shogi_model.keras\n")
        f.write("beta: 10.0\n")

with open("tasks.sh", "w") as f:
    for i in range(n_files):
        f.write(f".venv/bin/python main.py input{i:03d}.yaml\n")
