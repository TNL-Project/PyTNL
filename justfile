#!/usr/bin/env -S just --working-directory . --justfile

# Flags for tools that do not respect the FORCE_COLOR environment variable

color := if env("FORCE_COLOR", "") == "1" { "--color" } else { "" }
color_always := if env("FORCE_COLOR", "") == "1" { "--color always" } else { "" }

venv_bin := ".venv/bin"

# Note: The first recipe is run by default.

# Installs the project using pip (Default includes CUDA; pass "host" for CPU-only)
install variant="cuda":
    #!/usr/bin/env bash
    set -euo pipefail
    just _ensure-venv
    {{ venv_bin }}/pip install scikit-build-core
    variant="{{ variant }}"
    case "$variant" in
        cuda)
            {{ venv_bin }}/pip install --no-build-isolation -ve '.[dev,cuda]' \
                -C cmake.define.PyTNL_USE_CUDA=ON
            ;;
        host)
            {{ venv_bin }}/pip install --no-build-isolation -ve '.[dev]' \
                -C cmake.define.PyTNL_USE_CUDA=OFF
            ;;
        *)
            echo "Unknown variant '$variant'. Use 'cuda' (default) or 'host'." >&2
            exit 1
            ;;
    esac

# Runs all checks
check: check-format check-code check-typing check-typos check-recipes

# Builds an sdist tarball of the project using python-build
build-sdist:
    just _ensure-venv
    {{ venv_bin }}/python -m build --sdist --no-isolation

# Builds a wheel of the project using python-build
build-wheel:
    just _ensure-venv
    {{ venv_bin }}/python -m build --wheel --no-isolation

# Builds a wheel and an sdist tarball of the project using python-build
build: build-sdist build-wheel

# Cleans the build directory
clean:
    rm -frv dist build

# Checks the code using ruff
check-code:
    just _ensure-command ruff
    ruff check

# Checks the code formatting using clang-format, gersemi, and ruff
check-format:
    just --unstable --fmt --check
    just _ensure-command clang-format
    # Note that find -exec always exists with 0 exit code, whereas xargs runs
    # clang-format only once and preserves its exit code.
    find ./include/ ./src/ \
         \( -name '*.h' \
         -o -name '*.hpp' \
         -o -name '*.cpp' \
         -o -name '*.cu' \) \
         -print0 | xargs -0 clang-format --dry-run -Werror --style file
    just _ensure-command gersemi
    gersemi {{ color }} --diff --check .
    just _ensure-command ruff
    ruff format --diff

# Reformats supported files using clang-format and ruff
format:
    just --unstable --fmt
    just _ensure-command clang-format
    # Note that find -exec always exists with 0 exit code, whereas xargs runs
    # clang-format only once and preserves its exit code.
    find ./include/ ./src/ \
         \( -name '*.h' \
         -o -name '*.hpp' \
         -o -name '*.cpp' \
         -o -name '*.cu' \) \
         -print0 | xargs -0 clang-format -i --style file
    just _ensure-command gersemi
    gersemi {{ color }} --in-place .
    just _ensure-command ruff
    ruff format .

# Checks for typing issues using pyright and mypy
check-typing:
    just _ensure-venv
    {{ venv_bin }}/basedpyright
    {{ venv_bin }}/mypy

# Checks for common spelling mistakes using typos
check-typos:
    just _ensure-command typos
    typos {{ color_always }} --sort

# Checks justfile recipe for shell issues using shellcheck
check-recipe recipe:
    just _ensure-command grep shellcheck
    just -vv -n {{ recipe }} 2>&1 | grep -v '===> Running recipe' | shellcheck -

# Checks all justfile recipes with inline bash for shell issues using shellcheck
check-recipes:
    just check-recipe 'install'
    just check-recipe '_ensure-command command'
    just check-recipe '_ensure-venv'
    just check-recipe 'create-pypi-release'
    just check-recipe 'release'

# Runs all tests (extra args are forwarded to pytest, e.g. just test -m cuda -n 0)
test *args:
    just _ensure-venv
    {{ venv_bin }}/pytest {{ args }}

# Ensures that one or more required commands are installed
_ensure-command +command:
    #!/usr/bin/env bash
    set -euo pipefail

    read -r -a commands <<< "{{ command }}"

    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" > /dev/null 2>&1 ; then
            printf "Couldn't find required executable '%s'\n" "$cmd" >&2
            exit 1
        fi
    done

# Ensures that a virtual environment exists in .venv (run `just install` to populate it)
_ensure-venv:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
    fi

# Gets the project name from the pyproject.toml
get-project-name:
    just _ensure-command yq
    yq -r '.project.name' pyproject.toml

# Gets the current version of the project from the pyproject.toml
get-current-version:
    just _ensure-command yq
    yq -r '.project.version' pyproject.toml

# Builds a sdist tarball of the project and uploads it to PyPI using twine
create-pypi-release:
    #!/usr/bin/env bash
    set -euo pipefail

    project="$(just get-project-name)"
    readonly project="$project"
    if [[ -z "$project" ]]; then
        printf "No project name found!\n" >&2
        exit 1
    fi
    current_version="$(just get-current-version)"
    readonly current_version="$current_version"
    if [[ -z "$current_version" ]]; then
        printf "No current version found!\n" >&2
        exit 1
    fi

    just _ensure-command git twine # gpg glab

    if ! git tag --points-at | grep "$current_version" >/dev/null; then
        printf "Current project version is %s, but HEAD is not the tag %s!\n" "$current_version" "$current_version" >&2
        exit 1
    fi

    rm -fv dist/*
    just build-sdist

    #printf "Creating signature for sdist tarball...\n"
    #gpg -o "dist/$project-$current_version.tar.gz.sig" --default-key "$(git config --local --get user.signingkey)" -s "dist/$project-$current_version.tar.gz"
    #printf "Attach sdist tarball and signature to GitLab release %s...\n" "$current_version"
    #GITLAB_HOST=gitlab.archlinux.org glab release create "$current_version" "./dist/$project-$current_version.tar.gz"* --name="$current_version" --notes="$current_version"

    printf "Pushing the sdist tarball to PyPI...\n"
    twine upload "./dist/$project-$current_version.tar.gz"

# Creates a GitLab release using the glab CLI tool
create-gitlab-release:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command git

    # Check that we are on the main branch
    if [[ "$(git branch --show-current)" != "main" ]]; then
        printf "You are not on the main branch!\n" >&2
        exit 1
    fi

    # Pull the latest changes
    git pull --tags origin

    # Get the current version from the last git tag
    current_version="$(git tag --sort=version:refname | tail -n 1)"
    if [[ -z "$current_version" ]]; then
        printf "No current version found!\n" >&2
        exit 1
    fi

    # Get the previous version (if any)
    previous_version="$(git tag --sort=version:refname | tail -n 2 | head -n 1)"

    # Prepare initial release notes (can be edited manually on GitLab)
    release_notes="# Release notes - version $current_version\n\n"
    if [[ -n "$previous_version" ]]; then
        release_notes+="## Merge requests\n\n$(git log --pretty=format:"* %w(0,0,2)%b" --merges "$previous_version..$current_version")\n\n"
        release_notes+="## Detailed changes\n\n$(git log --pretty=format:"* %s (%H)" --no-merges "$previous_version..$current_version")\n\n"
    fi
    # Run through echo to interpret escapes such as \n
    release_notes="$(echo -e "$release_notes")"

    just _ensure-command glab

    printf "Creating GitLab release %s\n" "$current_version"
    glab release create "$current_version" --no-update --ref="$current_version" --name="$current_version" --notes="$release_notes"

# Creates a tag and pushes it (if it does exist yet) and creates a release for it
release:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command git

    # Check that we are on the main branch
    if [[ "$(git branch --show-current)" != "main" ]]; then
        printf "You are not on the main branch!\n" >&2
        exit 1
    fi

    # Pull the latest changes
    git pull --tags origin

    # Get the current version of the project
    current_version="$(just get-current-version)"
    readonly current_version="$current_version"
    if [[ -z "$current_version" ]]; then
        printf "No current version found!\n" >&2
        exit 1
    fi

    # Check that the tag does not exist yet
    if [[ -n "$(git tag -l "$current_version")" ]]; then
        printf "The tag %s exists already!\n" "$current_version" >&2
        exit 1
    fi

    # Create a new tag and push it
    git push origin
    printf "Creating tag %s...\n" "$current_version"
    git tag -a "$current_version" -m "version $current_version"
    printf "Pushing tag %s...\n" "$current_version"
    git push origin refs/tags/"$current_version"

    # Create a release on GitLab
    just create-gitlab-release

    # Create a release on PyPI
    just create-pypi-release
