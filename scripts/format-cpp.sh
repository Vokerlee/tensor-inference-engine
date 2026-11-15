#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
export PATH="$root/.venv/bin:$PATH"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format not found; install with: .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 127
fi

mapfile -d '' -t tracked < <(
    git ls-files -z -- \
        '*.cu' '*.cuh' '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' '*.ipp'
)

# git lists index entries, so a file deleted in the working tree is still reported.
files=()
for file in "${tracked[@]}"; do
    if [ -f "$file" ]; then
        files+=("$file")
    fi
done

if [ ${#files[@]} -eq 0 ]; then
    exit 0
fi

case "${1-}" in
    --check) exec clang-format --dry-run --Werror "${files[@]}" ;;
    *) exec clang-format -i "${files[@]}" ;;
esac
