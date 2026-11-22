# usage: python3 -I compare-servers.py BASE_URL_A BASE_URL_B
# Sends the same compilations (cache bypassed) to two CE instances and reports any difference in asm, stderr,
# execution output or tool output. Compilers, tools and libraries in CASES must exist on both; edit them to cover
# what your change touches.
import json
import sys
import urllib.request

PLAIN = '#include <cstdio>\nint square(int x) { return x * x; }\nint main() { std::printf("%d\\n", square(7)); }\n'
LIBRARY = '#include <cstdio>\n#include <gcem.hpp>\nint main() { std::printf("%d\\n", (int)gcem::sqrt(81.0)); }\n'
CMAKE = 'cmake_minimum_required(VERSION 3.5)\nproject(hello)\nadd_executable(output.s main.cpp)\n'
RUST = 'pub fn square(x: i32) -> i32 { x * x }\nfn main() { println!("{}", square(5)); }\n'
FILTERS = {'intel': True, 'labels': True, 'directives': True, 'commentOnly': True, 'demangle': True}


def options(user='-O2', filters=None, tools=(), libraries=(), execute=False):
    return {
        'userArguments': user,
        'compilerOptions': {},
        'filters': {**(filters or FILTERS), 'execute': execute},
        'tools': list(tools),
        'libraries': list(libraries),
        'executeParameters': {'args': [], 'stdin': ''},
    }


def tool(tool_id, args=''):
    return {'id': tool_id, 'args': args, 'stdin': ''}


# (name, compiler id, 'compile' or 'cmake', request body)
CASES = [
    ('asm', 'g161', 'compile', {'source': PLAIN, 'options': options()}),
    ('binary_execute', 'g161', 'compile',
     {'source': PLAIN, 'options': options(filters={**FILTERS, 'binary': True}, execute=True)}),
    ('tools', 'g161', 'compile', {'source': PLAIN, 'options': options(
        tools=[tool('llvm-mcatrunk'), tool('clangtidytrunk', '-checks=*'), tool('PVS-Studio')])}),
    ('library_execute', 'g161', 'compile', {'source': LIBRARY, 'options': options(
        libraries=[{'id': 'gcem', 'version': '1_17_0'}], execute=True, tools=[tool('PVS-Studio')])}),
    ('clang_tools', 'clang2110', 'compile',
     {'source': PLAIN, 'options': options(tools=[tool('llvm-mcatrunk'), tool('clangquerytrunk')])}),
    ('cmake_execute', 'clang2110', 'cmake', {
        'source': CMAKE, 'files': [{'filename': 'main.cpp', 'contents': PLAIN}], 'options': options(execute=True)}),
    ('rust_execute', 'r1910', 'compile',
     {'source': RUST, 'lang': 'rust', 'options': options(user='-C opt-level=2', execute=True)}),
]


def compile_on(base, compiler, kind, body):
    url = f'{base}/api/compiler/{compiler}/' + ('build/cmake' if kind == 'cmake' else 'compile')
    request = urllib.request.Request(
        url, json.dumps({**body, 'bypassCache': 1}).encode(),
        {'Content-Type': 'application/json', 'Accept': 'application/json'})
    return json.loads(urllib.request.urlopen(request, timeout=180).read())


def texts(lines):
    return [line.get('text') for line in lines or []]


def summarise(result):
    execution = result.get('execResult') or {}
    return {
        'code': result.get('code'),
        'asm': texts(result.get('asm')),
        'stderr': texts(result.get('stderr')),
        'execution': (execution.get('code'), texts(execution.get('stdout')), execution.get('didExecute')),
        'tools': {t.get('id'): (t.get('code'), texts(t.get('stdout')), texts(t.get('stderr')))
                  for t in result.get('tools', [])},
    }


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: compare-servers.py BASE_URL_A BASE_URL_B')
    differences = 0
    for name, compiler, kind, body in CASES:
        a = summarise(compile_on(sys.argv[1], compiler, kind, body))
        b = summarise(compile_on(sys.argv[2], compiler, kind, body))
        tool_codes = ', '.join(f'{tool_id}:{v[0]}' for tool_id, v in a['tools'].items())
        print(f'{name}: {"SAME" if a == b else "DIFFERENT"} code={a["code"]} asm_lines={len(a["asm"])} '
              f'execution={a["execution"][0]} {a["execution"][1][:2]} tools={{{tool_codes}}}')
        if a != b:
            differences += 1
            for field in a:
                if a[field] != b[field]:
                    print(f'  {field}:\n    A: {str(a[field])[:400]}\n    B: {str(b[field])[:400]}')
        if a['code'] != 0:
            print(f'  stderr (both servers): {a["stderr"][:5]}')
    sys.exit(1 if differences else 0)


if __name__ == '__main__':
    main()
