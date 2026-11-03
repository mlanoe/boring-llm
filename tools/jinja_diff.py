#!/usr/bin/env python3
"""Differential test of src/jinja.br against real jinja2.

Renders a corpus of small templates (one construct each, drawn from what real
chat templates use) with jinja2 and with `boring run test/jinja_driver.br`, and
reports every case where the two disagree: different output, or one side
raising an error the other does not. jinja2 is configured like the project's
other oracle (tools/oracle_render_request.py): a plain Environment with HF's
`tojson` and a `raise_exception` global.

Usage (repo root, project venv):
    .venv/bin/python tools/jinja_diff.py            # every case, prints failures
    .venv/bin/python tools/jinja_diff.py -v         # also lists the passing cases
    .venv/bin/python tools/jinja_diff.py --boring boring    # compiler executable
Exit status 0 when every case agrees, 1 otherwise.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import jinja2

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hello there"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "  padded  "},
]
CTX = {
    "messages": MESSAGES,
    "name": "World",
    "items": ["a", "b", "c"],
    "nums": [3, 1, 2],
    "empty": [],
    "flag": True,
    "off": False,
    "n": 7,
    "f": 2.5,
    "text": "Hello, World",
    "padded": "  x y  ",
    "mixed": "aBc",
    "tools": [{"name": "get_weather", "parameters": {"city": {"type": "string"}}}],
    "d": {"a": 1, "b": 2, "c": 3},
    "nested": {"outer": {"inner": "deep"}},
    "nothing": None,
    "bos_token": "<s>",
    "words": "one two three",
}

# (name, template) pairs; each template is rendered with CTX.
CASES = [
    # --- already-supported baseline
    ("output", "{{ name }}"),
    ("if/else", "{% if flag %}y{% else %}n{% endif %}"),
    ("for+loop", "{% for x in items %}{{ loop.index }}{{ x }}{% if not loop.last %},{% endif %}{% endfor %}"),
    ("set", "{% set x = 3 %}{{ x + 1 }}"),
    ("namespace", "{% set ns = namespace(a=0) %}{% for x in items %}{% set ns.a = ns.a + 1 %}{% endfor %}{{ ns.a }}"),
    ("trim", "{{ padded|trim }}"),
    ("length", "{{ items|length }}"),
    ("tojson", "{{ d|tojson }}"),
    ("slice", "{{ items[1:] }}"),
    ("neg index", "{{ items[-1] }}"),
    ("startswith", "{{ text.startswith('Hello') }}"),
    # --- operators
    ("tilde concat", "{{ 'a' ~ 1 ~ 'b' }}"),
    ("mul", "{{ n * 2 }}"),
    ("floordiv", "{{ n // 2 }}"),
    ("truediv", "{{ n / 2 }}"),
    ("mod", "{{ n % 3 }}"),
    ("pow", "{{ 2 ** 3 }}"),
    ("float literal", "{{ 1.5 + f }}"),
    ("string mul", "{{ 'ab' * 2 }}"),
    ("ternary", "{{ 'yes' if flag else 'no' }}"),
    ("ternary no else", "[{{ 'yes' if off }}]"),
    ("not in", "{{ 'z' not in items }}"),
    ("chained cmp", "{{ 1 < n < 10 }}"),
    ("is divisibleby", "{{ n is divisibleby 7 }}"),
    ("is odd", "{{ n is odd }}"),
    ("is even", "{{ n is even }}"),
    ("is in", "{{ 'a' is in items }}"),
    ("is eq", "{{ n is eq 7 }}"),
    ("is iterable", "{{ items is iterable }}"),
    ("is integer", "{{ n is integer }}"),
    ("is float", "{{ f is float }}"),
    # --- literals
    ("list literal", "{{ [1, 2, 3]|length }}"),
    ("list literal iter", "{% for x in ['p', 'q'] %}{{ x }}{% endfor %}"),
    ("dict literal", "{{ {'a': 1}['a'] }}"),
    ("tuple literal", "{% for a, b in [(1, 2), (3, 4)] %}{{ a + b }}{% endfor %}"),
    ("true/false lower", "{{ true }}{{ false }}{{ none }}"),
    ("None literal", "{{ None is none }}"),
    ("string escapes", "{{ 'a\\nb' }}"),
    ("double quotes", '{{ "it\'s" }}'),
    # --- filters
    ("upper", "{{ mixed|upper }}"),
    ("lower", "{{ mixed|lower }}"),
    ("title", "{{ words|title }}"),
    ("capitalize", "{{ words|capitalize }}"),
    ("first", "{{ items|first }}"),
    ("last", "{{ items|last }}"),
    ("join", "{{ items|join(', ') }}"),
    ("join nosep", "{{ items|join }}"),
    ("reverse", "{{ items|reverse|join }}"),
    ("sort", "{{ nums|sort|join(',') }}"),
    ("default", "{{ missing|default('dflt') }}"),
    ("default bool", "{{ off|default('dflt', true) }}"),
    ("d alias", "{{ missing|d('x') }}"),
    ("string filter", "{{ n|string + '!' }}"),
    ("int filter", "{{ '42'|int + 1 }}"),
    ("float filter", "{{ '1.5'|float }}"),
    ("replace filter", "{{ text|replace('World', 'There') }}"),
    ("trim chars", "{{ 'xxhixx'|trim('x') }}"),
    ("list filter", "{{ 'abc'|list|join('-') }}"),
    ("map attribute", "{{ messages|map(attribute='role')|join(',') }}"),
    ("selectattr", "{{ messages|selectattr('role', 'equalto', 'user')|list|length }}"),
    ("rejectattr", "{{ messages|rejectattr('role', 'equalto', 'user')|list|length }}"),
    ("select", "{{ nums|select('odd')|list|length }}"),
    ("length string", "{{ text|length }}"),
    ("count", "{{ items|count }}"),
    ("abs", "{{ (-3)|abs }}"),
    ("round", "{{ f|round }}"),
    ("sum", "{{ nums|sum }}"),
    ("min max", "{{ nums|min }}{{ nums|max }}"),
    ("unique", "{{ ['a','a','b']|unique|join }}"),
    ("items filter", "{% for k, v in d|dictsort %}{{ k }}{{ v }}{% endfor %}"),
    ("indent", "{{ 'a\\nb'|indent(2) }}"),
    ("wordcount", "{{ words|wordcount }}"),
    ("e/escape", "{{ '<b>'|e }}"),
    ("safe", "{{ '<b>'|safe }}"),
    ("tojson indent", "{{ d|tojson(indent=2) }}"),
    ("tojson list", "{{ items|tojson }}"),
    ("tojson str", "{{ text|tojson }}"),
    ("tojson none", "{{ nothing|tojson }}"),
    ("batch", "{{ items|batch(2)|list|length }}"),
    ("filter block", "{% filter upper %}abc{% endfilter %}"),
    # --- dict / list / string methods
    ("dict items", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}"),
    ("dict keys", "{{ d.keys()|list|join(',') }}"),
    ("dict values", "{{ d.values()|list|sum }}"),
    ("dict get", "{{ d.get('a') }}{{ d.get('zz', 'dflt') }}"),
    ("dict in", "{{ 'a' in d }}"),
    ("dict index", "{{ d['b'] }}"),
    ("nested attr", "{{ nested.outer.inner }}"),
    ("nested index", "{{ nested['outer']['inner'] }}"),
    ("str upper method", "{{ mixed.upper() }}"),
    ("str title method", "{{ words.title() }}"),
    ("str capitalize method", "{{ words.capitalize() }}"),
    ("str find", "{{ text.find('World') }}"),
    ("str count", "{{ text.count('l') }}"),
    ("str join", "{{ ', '.join(items) }}"),
    ("str format", "{{ '%s-%s'|format('a', 'b') }}"),
    ("str split default", "{{ words.split()|length }}"),
    ("str split max", "{{ words.split(' ', 1)|length }}"),
    ("str rsplit", "{{ words.rsplit(' ', 1)[0] }}"),
    ("str isdigit", "{{ '12'.isdigit() }}"),
    ("str index slice", "{{ text[0] }}{{ text[-1] }}{{ text[:5] }}"),
    ("str step slice", "{{ items[::-1]|join }}"),
    ("list step slice", "{{ items[::2]|join }}"),
    ("list index negative slice", "{{ items[:-1]|join }}"),
    ("list concat", "{{ (items + ['d'])|length }}"),
    ("list in for", "{% for m in messages[1:] %}{{ m.role[0] }}{% endfor %}"),
    ("list append", "{% set l = [] %}{% set _ = l.append(1) %}{{ l|length }}"),
    # --- statements
    ("for else", "{% for x in empty %}{{ x }}{% else %}none{% endfor %}"),
    ("for if filter", "{% for x in items if x != 'b' %}{{ x }}{% endfor %}"),
    ("for unpack dict", "{% for k, v in d.items() %}{{ k }}{% endfor %}"),
    ("for tuple unpack enumerate-like", "{% for p in items %}{{ loop.index0 }}{{ p }}{% endfor %}"),
    ("loop.revindex", "{% for x in items %}{{ loop.revindex }}{% endfor %}"),
    ("loop.revindex0", "{% for x in items %}{{ loop.revindex0 }}{% endfor %}"),
    ("loop.length", "{% for x in items %}{{ loop.length }}{% endfor %}"),
    ("loop.cycle", "{% for x in items %}{{ loop.cycle('o', 'e') }}{% endfor %}"),
    ("loop.previtem", "{% for x in items %}{{ loop.previtem }}{% endfor %}"),
    ("loop.nextitem", "{% for x in items %}{{ loop.nextitem }}{% endfor %}"),
    ("nested loops", "{% for x in items %}{% for y in items %}{{ x }}{{ y }}{% endfor %}{% endfor %}"),
    ("loop outer ref", "{% for x in items %}{% set o = loop %}{% for y in items %}{{ o.index }}{% endfor %}{% endfor %}"),
    ("elif chain", "{% if n == 1 %}a{% elif n == 7 %}b{% else %}c{% endif %}"),
    ("set multiple", "{% set a, b = 1, 2 %}{{ a + b }}"),
    ("set block", "{% set x %}captured{% endset %}{{ x }}"),
    ("set list", "{% set l = [1, 2] %}{{ l|length }}"),
    ("set dict", "{% set m = {'k': 'v'} %}{{ m.k }}"),
    ("macro", "{% macro hi(who) %}Hi {{ who }}{% endmacro %}{{ hi('A') }}"),
    ("macro default arg", "{% macro hi(who='X') %}Hi {{ who }}{% endmacro %}{{ hi() }}"),
    ("macro kwargs", "{% macro hi(a, b=2) %}{{ a }}{{ b }}{% endmacro %}{{ hi(1, b=3) }}"),
    ("call block", "{% macro w() %}[{{ caller() }}]{% endmacro %}{% call w() %}in{% endcall %}"),
    ("comment", "a{# hidden #}b"),
    ("raw", "{% raw %}{{ x }}{% endraw %}"),
    ("break", "{% for x in items %}{% if x == 'b' %}{% break %}{% endif %}{{ x }}{% endfor %}"),
    ("continue", "{% for x in items %}{% if x == 'b' %}{% continue %}{% endif %}{{ x }}{% endfor %}"),
    ("raise_exception", "{{ raise_exception('boom') }}"),
    ("raise in if", "{% if off %}{{ raise_exception('boom') }}{% endif %}ok"),
    ("whitespace both", "a  {%- if flag -%}  b  {%- endif -%}  c"),
    ("trim only left", "a  {{- name }}"),
    ("trim only right", "{{ name -}}  c"),
    ("nested if in for", "{% for m in messages %}{% if m.role == 'user' %}U{% elif m.role == 'assistant' %}A{% else %}S{% endif %}{% endfor %}"),
    ("namespace multi", "{% set ns = namespace(a=0, b='x') %}{% set ns.b = ns.b ~ 'y' %}{{ ns.b }}"),
    ("namespace in loop cond", "{% set ns = namespace(found=false) %}{% for m in messages %}{% if m.role == 'user' %}{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}"),
    ("range", "{% for i in range(3) %}{{ i }}{% endfor %}"),
    ("range 2 args", "{% for i in range(1, 4) %}{{ i }}{% endfor %}"),
    ("range step", "{% for i in range(0, 6, 2) %}{{ i }}{% endfor %}"),
    ("dict function", "{{ dict(a=1)['a'] }}"),
    ("lipsum-free cycler", "{% set c = cycler('a', 'b') %}{{ c.next() }}{{ c.next() }}{{ c.next() }}"),
    ("strftime_now", "{{ strftime_now('%Y') is string }}"),
    ("with block", "{% with x = 5 %}{{ x }}{% endwith %}"),
    ("is mapping", "{{ d is mapping }}"),
    ("is string on number", "{{ n is string }}"),
    ("is sequence", "{{ items is sequence }}"),
    ("is number", "{{ f is number }}"),
    ("is defined", "{{ missing is defined }}{{ name is defined }}"),
    ("is none", "{{ nothing is none }}"),
    ("is not none", "{{ name is not none }}"),
    ("is callable", "{{ name is callable }}"),
    # --- chat-template idioms
    ("system first", "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% endif %}"),
    ("content trim", "{{ messages[-1]['content']|trim }}"),
    ("bos var", "{{ bos_token }}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"),
    ("alternating check", "{% for m in messages %}{% if (m['role'] == 'user') != (loop.index0 % 2 == (1 if messages[0]['role'] == 'system' else 0)) %}X{% endif %}{% endfor %}ok"),
    ("tools loop", "{% for t in tools %}{{ t.name }}:{{ t.parameters|tojson }}{% endfor %}"),
    ("tools length", "{% if tools is defined and tools|length > 0 %}T{% endif %}"),
    ("message content string check", "{% for m in messages %}{% if m['content'] is string %}s{% else %}o{% endif %}{% endfor %}"),
    ("add_generation_prompt undefined", "{% if add_generation_prompt %}G{% endif %}end"),
    ("last message role", "{{ messages[-1].role }}"),
    ("selectattr first", "{{ (messages|selectattr('role', 'equalto', 'system')|first).content }}"),
    ("reverse loop", "{% for m in messages|reverse %}{{ m.role[0] }}{% endfor %}"),
    ("loop.last with if", "{% for m in messages %}{{ m.role }}{% if loop.last %}!{% endif %}{% endfor %}"),
    ("string comparison", "{{ 'a' < 'b' }}"),
    ("is not defined", "{% if messages[0].foo is not defined %}nofoo{% endif %}"),
    ("undefined attr str", "[{{ messages[0].foo }}]"),
    ("undefined var str", "[{{ nope }}]"),
    ("print int float", "{{ 1.0 }}{{ 3 / 2 }}{{ 4 / 2 }}"),
    ("bool print", "{{ true }} {{ false }}"),
    ("none print", "{{ none }}"),
    ("list print", "{{ items }}"),
    ("dict print", "{{ d }}"),
]


def reference(template):
    env = jinja2.Environment()
    env.filters["tojson"] = lambda x, indent=None, **kw: json.dumps(x, ensure_ascii=False, indent=indent)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: __import__("datetime").datetime.now().strftime(fmt)
    try:
        return True, env.from_string(template).render(**CTX)
    except Exception as e:  # noqa: BLE001 - any jinja2 failure counts as "error"
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true", help="also list the cases that agree")
    ap.add_argument("--boring", default="boring")
    ap.add_argument("--filter", help="only cases whose name contains this text")
    args = ap.parse_args()

    cases = [(n, t) for n, t in CASES if not args.filter or args.filter in n]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cases.json")
        with open(path, "w") as f:
            json.dump({"cases": [{"template": t, "context": CTX} for _, t in cases]}, f)
        run = subprocess.run(
            [args.boring, "run", os.path.join(root, "test", "jinja_driver.br"), "--", path],
            capture_output=True, text=True, cwd=root, stdin=subprocess.DEVNULL,
        )
    last = [ln for ln in run.stdout.splitlines() if ln.startswith("[")]
    if not last:
        print("driver produced no result line:\n" + (run.stdout + run.stderr)[-800:])
        return 1
    ours = json.loads(last[-1])
    if len(ours) != len(cases):
        print(f"driver returned {len(ours)} results for {len(cases)} cases")
        return 1

    bad = 0
    for (name, template), got in zip(cases, ours):
        ok, want = reference(template)
        if ok and got["ok"] and got["out"] == want:
            if args.v:
                print(f"ok    {name}")
            continue
        if not ok and not got["ok"]:
            if args.v:
                print(f"ok    {name} (both raise)")
            continue
        bad += 1
        if ok and not got["ok"]:
            print(f"FAIL  {name}: boring raised, jinja2 gives {want!r}")
        elif ok:
            print(f"FAIL  {name}: jinja2 {want!r}, boring {got['out']!r}")
        else:
            print(f"FAIL  {name}: jinja2 raises ({want[:60]}), boring gives {got.get('out')!r}")
    print(f"{len(cases) - bad}/{len(cases)} agree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
