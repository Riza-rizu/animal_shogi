$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
& .venv\Scripts\python.exe battle.py $args
