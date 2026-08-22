#!/usr/bin/env bash
# find_orphans.sh — audits every .py file in the repo for real importers.
# Usage:
#   bash find_orphans.sh            # dry run — just report
#   bash find_orphans.sh --delete   # after reviewing, actually git rm the orphans

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-report}"

# ── Files that legitimately have zero importers — CLI entry points, tests,
#    package markers, dynamic-import targets. Extend this as needed. ──
ALWAYS_KEEP_REGEX='(^|/)(__init__\.py|main\.py|main_coprocessor\.py|run_live\.py|run_real_telemetry\.py|sanity_checks\.py|jax_config\.py)$'
ALWAYS_KEEP_DIR_REGEX='^(tests?/|scripts/)'   # scripts/ and tests/ are mostly standalone CLI entry points

ORPHANS=()
KEPT_CLI=()

echo "Scanning for .py files with no real importers..."
echo "──────────────────────────────────────────────────────────────────"

while IFS= read -r -d '' file; do
    rel="${file#./}"

    # Skip package markers, known entry points
    if [[ "$rel" =~ $ALWAYS_KEEP_REGEX ]]; then
        continue
    fi

    base="$(basename "$rel" .py)"
    # Dotted module path, e.g. models/aero_platform.py -> models.aero_platform
    dotted="${rel%.py}"
    dotted="${dotted//\//.}"

    # Search for real import statements referencing this module, anywhere
    # in the repo, excluding the file itself.
    #   from X import ...
    #   from X.Y import ...   (dotted, e.g. from models.aero_platform import Foo)
    #   import X
    #   import X.Y
    hits=$(grep -rlE "^\s*(from[[:space:]]+(\.*[A-Za-z0-9_.]*\.)?${base}[[:space:]]+import|import[[:space:]]+(\.*[A-Za-z0-9_.]*\.)?${base}([[:space:]]|$)|from[[:space:]]+${dotted//./\\.}[[:space:]]+import|import[[:space:]]+${dotted//./\\.})" \
        --include="*.py" . 2>/dev/null | grep -v -- "^\./${rel}$" || true)

    if [[ -z "$hits" ]]; then
        if [[ "$rel" =~ $ALWAYS_KEEP_DIR_REGEX ]]; then
            KEPT_CLI+=("$rel")
        else
            ORPHANS+=("$rel")
        fi
    fi
done < <(find . -name "*.py" -not -path "./.git/*" -print0)

echo ""
echo "═══ Standalone CLI tools (scripts/ or tests/, zero importers = expected) ═══"
printf '  %s\n' "${KEPT_CLI[@]}"

echo ""
echo "═══ ORPHAN CANDIDATES (no importers found, not a known entry point) ═══"
if [[ ${#ORPHANS[@]} -eq 0 ]]; then
    echo "  (none found)"
else
    printf '  %s\n' "${ORPHANS[@]}"
fi

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo "Reviewed ${#ORPHANS[@]} orphan candidates, ${#KEPT_CLI[@]} standalone tools skipped."

if [[ "$MODE" == "--delete" ]]; then
    if [[ ${#ORPHANS[@]} -eq 0 ]]; then
        echo "Nothing to delete."
        exit 0
    fi
    echo ""
    echo "About to git rm the ${#ORPHANS[@]} orphan candidates listed above."
    read -rp "Type 'yes' to confirm: " confirm
    if [[ "$confirm" == "yes" ]]; then
        git rm "${ORPHANS[@]}"
        echo "Deleted. Review with 'git status' and commit when ready."
    else
        echo "Aborted — nothing deleted."
    fi
else
    echo ""
    echo "This was a dry run. Review the ORPHAN CANDIDATES list carefully —"
    echo "grep-based import detection cannot catch dynamic imports (importlib,"
    echo "string-based module loading, YAML/config-driven dispatch)."
    echo "Re-run with --delete once you've eyeballed the list."
fi