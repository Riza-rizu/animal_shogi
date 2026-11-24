param(
    [int]$StartGen = 1,
    [int]$EndGen = 5,
    [string]$System = "quantum_d0_perm9"
)

# スクリプトの配置場所から相対的にパスを決定（日本語文字化けを完全防止）
$base_local = Join-Path $PSScriptRoot "experiments\$System"
$server_base = "imai@watanabe-login.appi.keio.ac.jp:github/animal_shogi/quantum_animal_shogi/experiments/$System"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Download: Gen$StartGen to Gen$EndGen" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

for ($g = $StartGen; $g -le $EndGen; $g++) {
    $p = $g - 1
    $target_dir = Join-Path $base_local "gen$g"
    if (!(Test-Path $target_dir)) {
        New-Item -ItemType Directory -Force -Path $target_dir | Out-Null
    }

    Write-Host "`n>>> Downloading Gen$g (1回の接続で10ファイル一括取得) ..." -ForegroundColor Cyan

    $files = @(
        "quantum_model_gen$g.keras",
        "scatter_evaluation.png",
        "moves_distribution_gen$g.png",
        "entropy_decay_curve_gen$g.png",
        "entropy_metrics_gen$g.json",
        "metrics.json",
        "training_history.json",
        "match_stats_gen$p.json",
        "match_summary_1million_games_gen$p.csv",
        "match_replay_gen$p.npz"
    )

    $remote_args = @()
    foreach ($f in $files) {
        $remote_args += "${server_base}/gen$g/$f"
    }

    # 1つのscpコマンドに全10ファイルを渡すことで、パスフレーズ入力を「1回だけ」に集約
    scp @remote_args "$target_dir"
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "Download complete: Gen$StartGen to Gen$EndGen" -ForegroundColor Green
Write-Host "Destination: $base_local" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green