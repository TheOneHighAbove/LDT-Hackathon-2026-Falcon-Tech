[CmdletBinding()]
param(
    [switch] $Gpu,
    [switch] $Build,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $InferenceArguments
)

$ErrorActionPreference = "Stop"
$repositoryDirectory = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is required to run containerized inference"
}
if (-not $Gpu) {
    throw "score-optimized inference requires CUDA; rerun with -Gpu"
}

Push-Location $repositoryDirectory
try {
    New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
    $composeArguments = @("compose", "-f", "docker-compose.yml")
    $composeArguments += @("run", "--rm")
    if ($Build) {
        $composeArguments += "--build"
    }
    $composeArguments += @(
        "--no-deps", "reid-api",
        "python", "-m", "scripts.infer_score_optimized",
        "--release-config", "/app/configs/score_optimized_speed.json",
        "--weights-dir", "/app/weights",
        "--images-dir", "/app/dataset/images",
        "--query-csv", "/app/dataset/test_query.csv",
        "--gallery-csv", "/app/dataset/test_gallery.csv",
        "--output-dir", "/app/outputs/score_optimized_speed"
    )
    & docker @composeArguments @InferenceArguments
    if ($LASTEXITCODE -ne 0) {
        throw "containerized inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
