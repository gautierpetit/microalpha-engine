$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Remove previously copied extension from package dir if possible
$existingExtensions = Get-ChildItem ".\microalpha\_cpp*.pyd" -ErrorAction SilentlyContinue
foreach ($file in $existingExtensions) {
    try {
        Remove-Item $file.FullName -Force
        Write-Host "Removed old extension: $($file.Name)"
    }
    catch {
        Write-Error "Cannot remove $($file.FullName). The extension is likely loaded by a running Python process or VS Code kernel. Close all Python sessions and try again."
        exit 1
    }
}

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