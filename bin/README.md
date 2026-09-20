Place your compiled audiocpp_cli binary here (or ensure it is on PATH).

Build only the yue2 family, e.g.:
  Linux:  ./scripts/build_linux.sh --backend cpu --model-set custom --models yue2 --target audiocpp_cli
  macOS:  ./scripts/build_metal.sh --model-set custom --models yue2 --target audiocpp_cli
  Windows (PowerShell):
    .\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models "yue2" -Target audiocpp_cli
