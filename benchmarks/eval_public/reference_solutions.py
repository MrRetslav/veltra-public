"""Reference solutions, one per entry of code_tasks.CODE_TASKS, in the same order. Used only by `run_eval.py --validate` to prove that every
hidden test is satisfiable; never sent to a model."""

REFERENCE = [
'''def rle_encode(s):
    import itertools
    return ''.join(str(len(list(g))) + k for k, g in itertools.groupby(s))''',
'''def merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out''',
'''def roman_to_int(s):
    v = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    return sum(-v[c] if i + 1 < len(s) and v[c] < v[s[i + 1]] else v[c] for i, c in enumerate(s))''',
'''def balanced(s):
    st, pair = [], {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in '([{': st.append(c)
        elif c in pair:
            if not st or st.pop() != pair[c]: return False
    return not st''',
'''def second_largest(n):
    u = sorted(set(n))
    return u[-2] if len(u) > 1 else None''',
'''def caesar(s, k):
    r = []
    for c in s:
        if c.isascii() and c.isalpha():
            b = ord('a') if c.islower() else ord('A')
            c = chr(b + (ord(c) - b + k) % 26)
        r.append(c)
    return ''.join(r)''',
'''def flatten(x):
    out = []
    for i in x:
        if isinstance(i, list): out += flatten(i)
        else: out.append(i)
    return out''',
'''def top_words(text, n):
    import re, collections
    c = collections.Counter(w.lower() for w in re.findall('[A-Za-z]+', text))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]''',
'''def is_valid_ipv4(s):
    import re
    p = s.split('.')
    return len(p) == 4 and all(re.fullmatch('0|[1-9][0-9]{0,2}', x) and int(x) < 256 for x in p)''',
'''def transpose(m):
    return [list(r) for r in zip(*m)]''',
'''def group_anagrams(words):
    d = {}
    for w in words: d.setdefault(''.join(sorted(w)), []).append(w)
    return list(d.values())''',
'''def business_days(start, end):
    import datetime as D
    a, b = D.date.fromisoformat(start), D.date.fromisoformat(end)
    return sum((a + D.timedelta(i)).weekday() < 5 for i in range((b - a).days + 1))''',
'''def to_snake(name):
    import re
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\\1_\\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s).lower()''',
'''def chunk(lst, size):
    if size < 1: raise ValueError
    return [lst[i:i + size] for i in range(0, len(lst), size)]''',
'''def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in ('+', '-', '*', '/'):
            b, a = st.pop(), st.pop()
            if t == '+': st.append(a + b)
            elif t == '-': st.append(a - b)
            elif t == '*': st.append(a * b)
            else:
                q = abs(a) // abs(b)
                st.append(q if (a >= 0) == (b > 0) else -q)
        else: st.append(int(t))
    return st[-1]''',
'''def longest_unique_substring(s):
    last, start, best = {}, 0, 0
    for i, c in enumerate(s):
        if c in last and last[c] >= start: start = last[c] + 1
        last[c] = i
        best = max(best, i - start + 1)
    return best''',
'''def prime_factors(n):
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d); n //= d
        d += 1
    return out + ([n] if n > 1 else [])''',
'''def rotate(lst, k):
    if not lst: return []
    k %= len(lst)
    return lst[len(lst) - k:] + lst[:len(lst) - k]''',
'''def parse_duration(s):
    import re
    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', s)
    if not s or not m: raise ValueError
    h, mi, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + sec''',
'''def matrix_spiral(m):
    out = []
    while m:
        out += m[0]
        m = [list(r) for r in zip(*m[1:])][::-1]
    return out''',
'''def two_sum_pairs(nums, target):
    s = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target: s.add(tuple(sorted((nums[i], nums[j]))))
    return sorted(s)''',
'''def binary_search_first(a, x):
    import bisect
    i = bisect.bisect_left(a, x)
    return i if i < len(a) and a[i] == x else -1''',
'''def compress_ranges(nums):
    u, out, i = sorted(set(nums)), [], 0
    while i < len(u):
        j = i
        while j + 1 < len(u) and u[j + 1] == u[j] + 1: j += 1
        out.append(str(u[i]) if i == j else '%d-%d' % (u[i], u[j]))
        i = j + 1
    return ','.join(out)''',
'''def count_islands(grid):
    seen, n = set(), 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1' and (r, c) not in seen:
                n += 1
                st = [(r, c)]
                while st:
                    y, x = st.pop()
                    if (y, x) in seen or not (0 <= y < len(grid) and 0 <= x < len(grid[0])) or grid[y][x] != '1': continue
                    seen.add((y, x))
                    st += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return n''',
'''def lru_simulate(capacity, ops):
    from collections import OrderedDict
    d, out = OrderedDict(), []
    for op in ops:
        if op[0] == 'get':
            k = op[1]
            if k in d:
                d.move_to_end(k); out.append(d[k])
            else: out.append(-1)
        else:
            _, k, v = op
            d[k] = v; d.move_to_end(k)
            if len(d) > capacity: d.popitem(last=False)
    return out''',
'''def is_palindrome_sentence(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]''',
'''def topo_sort(n, edges):
    import heapq
    adj, deg = [[] for _ in range(n)], [0] * n
    for a, b in edges:
        adj[a].append(b); deg[b] += 1
    h = [i for i in range(n) if not deg[i]]
    heapq.heapify(h)
    out = []
    while h:
        x = heapq.heappop(h); out.append(x)
        for y in adj[x]:
            deg[y] -= 1
            if not deg[y]: heapq.heappush(h, y)
    return out if len(out) == n else None''',
'''def format_bytes(n):
    if n < 1024: return '%d B' % n
    v = n
    for u in ('KiB', 'MiB', 'GiB'):
        v /= 1024
        if v < 1024 or u == 'GiB': return '%.1f %s' % (v, u)''',
'''def dedupe_keep_order(items):
    out = []
    for i in items:
        if i not in out: out.append(i)
    return out''',
'''def word_wrap(text, width):
    out, cur = [], ''
    for w in text.split():
        if not cur: cur = w
        elif len(cur) + 1 + len(w) <= width: cur += ' ' + w
        else:
            out.append(cur); cur = w
    return out + ([cur] if cur else [])''',
'''def median(nums):
    s = sorted(nums); n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2''',
'''def next_permutation(nums):
    a = list(nums)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]: i -= 1
    if i < 0: return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]: j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = a[i + 1:][::-1]
    return a''',
'''def add_binary(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]''',
'''def overlap_minutes(a, b):
    m = lambda t: int(t[:2]) * 60 + int(t[3:])
    return max(0, min(m(a[1]), m(b[1])) - max(m(a[0]), m(b[0])))''',
'''def invert_index(docs):
    out = {}
    for i, t in docs.items():
        for w in set(t.lower().split()): out.setdefault(w, []).append(i)
    return {k: sorted(v) for k, v in out.items()}''',
'''def gcd_list(nums):
    import math
    return math.gcd(*nums) if len(nums) > 1 else abs(nums[0])''',
'''def decode_string(s):
    import re
    while '[' in s:
        s = re.sub(r'(\\d+)\\[([^\\[\\]]*)\\]', lambda m: m.group(2) * int(m.group(1)), s)
    return s''',
'''def moving_average(values, window):
    if window < 1: raise ValueError
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]''',
'''def hamming_distance(a, b):
    return bin(a ^ b).count('1')''',
'''def deep_merge(a, b):
    import copy
    r = copy.deepcopy(a)
    for k, v in b.items():
        r[k] = deep_merge(r[k], v) if isinstance(r.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return r''',
]
