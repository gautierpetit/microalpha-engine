$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pybindDir = & $venvPython -c "import pybind11; print(pybind11.get_cmake_dir())"

if (Test-Path build) {
    Remove-Item -Recurse -Force build
}

cmake -S cpp -B build `
    -DCMAKE_BUILD_TYPE=Release `
    -DPython_EXECUTABLE="$venvPython" `
    -Dpybind11_DIR="$pybindDir"

cmake --build build --config Release