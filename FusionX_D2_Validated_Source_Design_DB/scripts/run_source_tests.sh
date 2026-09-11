#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";cd "$ROOT";mkdir -p build reports build/compiled
run(){ name="$1";shift;echo "== $name ==";"$@" 2>&1 | tee "reports/${name}.log"; }
run generate_abi python3 tools/gen_abi.py
run abi_consistency python3 tools/check_abi_consistency.py
run structural_tokens python3 tools/sv_structural_check.py
run audit_contract python3 tools/audit_contract_check.py
run python_reference bash -lc 'cd sim/python && python3 test_reference.py'
run compiler_unit bash -lc 'cd compiler && python3 test_lowering.py'
run compiler_profiles python3 compiler/lowering.py
run c_runtime_build cc -std=c11 -Wall -Wextra -Werror -Isoftware/include software/runtime/fusionx_d2_runtime.c software/runtime/test_runtime.c -o build/test_runtime
run c_runtime_exec ./build/test_runtime
run firmware_syntax cc -std=c11 -Wall -Wextra -Werror -Isoftware/include -fsyntax-only software/firmware/boot_rom.c
run cpp_numeric_build c++ -std=c++20 -Wall -Wextra -Werror -Isim/cpp sim/cpp/fusionx_d2_numeric.cpp sim/cpp/test_numeric.cpp -o build/test_numeric
run cpp_numeric_exec ./build/test_numeric
printf 'PASS\n' > reports/source_tests.status
echo 'FUSIONX_D2_SOURCE_TESTS_PASS'
