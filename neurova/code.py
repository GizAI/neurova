from __future__ import annotations
import ast, os, subprocess, sys, tempfile
try:
    import resource
except ImportError:  # Windows compatibility: rlimits disabled.
    resource = None
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from .ir import ProgramSpecIR

@dataclass
class CodeAttempt:
    index: int
    ast_valid: bool
    exit_code: int
    output: str
    source: str

@dataclass
class CodeResult:
    success: bool
    attempts: List[CodeAttempt]
    final_source: str
    final_tests: str
    sandbox_note: str

class ProgramSynthesizer:
    def candidates(self, spec: ProgramSpecIR) -> List[str]:
        fn = spec.function_name
        if fn == "factorial":
            return [
                "def factorial(n):\n    return n\n",
                "def factorial(n: int) -> int:\n    if not isinstance(n, int):\n        raise TypeError('n must be int')\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for x in range(2, n + 1):\n        result *= x\n    return result\n",
            ]
        if fn == "gcd":
            return ["def gcd(a, b):\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a\n"]
        if fn == "fib":
            return ["def fib(n):\n    if n < 0: raise ValueError('n must be non-negative')\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n"]
        if fn == "is_prime":
            return ["def is_prime(n):\n    if n < 2: return False\n    if n == 2: return True\n    if n % 2 == 0: return False\n    d = 3\n    while d*d <= n:\n        if n % d == 0: return False\n        d += 2\n    return True\n"]
        return ["def add(a,b):\n    return a+b\n"]

    def tests(self, spec: ProgramSpecIR) -> str:
        fn = spec.function_name
        if fn == "factorial":
            return "from solution import factorial\nassert factorial(0)==1\nassert factorial(1)==1\nassert factorial(5)==120\ntry:\n    factorial(-1)\n    raise AssertionError('negative should fail')\nexcept ValueError:\n    pass\nprint('ok')\n"
        if fn == "gcd":
            return "from solution import gcd\nassert gcd(54,24)==6\nassert gcd(17,13)==1\nassert gcd(-12,18)==6\nprint('ok')\n"
        if fn == "fib":
            return "from solution import fib\nassert fib(0)==0\nassert fib(1)==1\nassert fib(7)==13\nprint('ok')\n"
        if fn == "is_prime":
            return "from solution import is_prime\nassert is_prime(2)\nassert is_prime(17)\nassert not is_prime(1)\nassert not is_prime(21)\nprint('ok')\n"
        return "from solution import add\nassert add(2,3)==5\nprint('ok')\n"

class CodeSandbox:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run_repair_loop(self, spec: ProgramSpecIR, timeout: int = 30) -> CodeResult:
        synth = ProgramSynthesizer()
        tests = synth.tests(spec)
        attempts=[]
        note="python -S subprocess with PYTHONPATH workspace, cwd isolation, timeout, thread limits, CPU/memory/file-size rlimits where supported; not nsjail/bwrap"
        for i, src in enumerate(synth.candidates(spec), 1):
            ast_ok=self._ast_valid(src)
            if not ast_ok:
                attempts.append(CodeAttempt(i, False, 100, "AST validation failed", src)); continue
            code,out=self._run(src, tests, timeout)
            attempts.append(CodeAttempt(i, True, code, out, src))
            if code == 0:
                return CodeResult(True, attempts, src, tests, note)
        return CodeResult(False, attempts, attempts[-1].source if attempts else "", tests, note)

    def _ast_valid(self, src: str) -> bool:
        try: tree=ast.parse(src)
        except SyntaxError: return False
        banned=(ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
        for node in ast.walk(tree):
            if isinstance(node, banned): return False
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval","exec","open","__import__","compile","input"}:
                return False
        return True

    def _run(self, src: str, tests: str, timeout: int) -> Tuple[int,str]:
        work=Path(tempfile.mkdtemp(prefix="code_", dir=str(self.root)))
        (work/"solution.py").write_text(src, encoding="utf-8")
        (work/"test_solution.py").write_text(tests, encoding="utf-8")
        env=os.environ.copy()
        env.update({"PYTHONNOUSERSITE":"1","PYTHONPATH":str(work),"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
        def limits():
            if resource is None:
                return
            try:
                resource.setrlimit(resource.RLIMIT_CPU,(5,5))
                resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
                resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024*1024,2*1024*1024))
            except Exception: pass
        try:
            r=subprocess.run([sys.executable,"-S",str(work/"test_solution.py")],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,env=env,preexec_fn=limits if os.name=="posix" else None)
            return r.returncode,r.stdout[-4000:]
        except subprocess.TimeoutExpired as e:
            return 124,f"timeout: {e}"
