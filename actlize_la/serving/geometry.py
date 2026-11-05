"""Small, host-only PLE geometry; never allocate a table, plan or device tensor.

SplitMix64/prime ordering follows b12x 560fef1 (Apache-2.0; see NOTICE).
"""

from dataclasses import dataclass

MASK64 = (1 << 64) - 1
I64_MAX = (1 << 63) - 1
GAMMA = 0x9E3779B97F4A7C15


def splitmix64(value):
    value = (value + GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime(value):
    if type(value) is not int or not 2 <= value <= I64_MAX:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    d, shifts = value - 1, 0
    while d % 2 == 0:
        d //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, d, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class PLEGeometry:
    prime_sizes: tuple[int, ...]
    table_offsets: tuple[int, ...]
    multipliers: tuple[int, ...]
    table_vocab_size: int
    padded_vocab_size: int
    vocab_size: int

    @property
    def nbytes(self):
        return (16 + 16 + 3) * 8


def ple_geometry(
    *,
    vocab_size,
    base_size=None,
    dense_layer_ordinal=0,
    alignment=128,
    prime_sizes=None,
    table_offsets=None,
    multipliers=None,
):
    """Prefer explicit checkpoint integer arrays. Fixed order3 / 8 heads/order.

    Safe product domain: token in [0,vocab_size), odd multiplier <= I64_MAX/vocab.
    Geometry is plain immutable data and can be materialized by any framework.
    """
    if (
        type(vocab_size) is not int
        or not 0 < vocab_size <= I64_MAX // 3
        or type(dense_layer_ordinal) is not int
        or dense_layer_ordinal < 0
        or type(alignment) is not int
        or not 0 < alignment <= I64_MAX
    ):
        raise ValueError("invalid integer PLE geometry")
    if (prime_sizes is None) != (table_offsets is None):
        raise ValueError("prime sizes and offsets must be supplied together")
    if prime_sizes is None:
        if type(base_size) is not int or not 0 < base_size < I64_MAX:
            raise ValueError("base_size required for geometry reconstruction")
        primes, candidate = [], base_size
        for i in range(16 * (dense_layer_ordinal + 1)):
            while not is_prime(candidate):
                candidate += 1
                if candidate > I64_MAX:
                    raise ValueError("prime geometry overflow")
            if i >= 16 * dense_layer_ordinal:
                primes.append(candidate)
            candidate += 1
        prime_sizes = primes
        table_offsets, total = [], 0
        for size in primes:
            table_offsets.append(total)
            total += size
    if multipliers is None:
        half_bound = max(1, (I64_MAX // vocab_size) // 2)
        multipliers = [
            2
            * (
                splitmix64(1234 + 10007 * dense_layer_ordinal + GAMMA * (i + 1))
                % half_bound
            )
            + 1
            for i in range(3)
        ]
    values = tuple(tuple(v) for v in (prime_sizes, table_offsets, multipliers))
    if tuple(map(len, values)) != (16, 16, 3) or any(
        type(x) is not int for v in values for x in v
    ):
        raise ValueError(
            "checkpoint geometry requires 16 primes/offsets and 3 integer multipliers"
        )
    primes, offsets, factors = values
    total = 0
    for size, offset in zip(primes, offsets):
        if not is_prime(size) or offset != total:
            raise ValueError("invalid prime or noncumulative offset")
        total += size
        if total > I64_MAX:
            raise ValueError("table extent overflow")
    if any(m <= 0 or m % 2 != 1 or m > I64_MAX // vocab_size for m in factors):
        raise ValueError("unsafe multiplier/token product")
    padded = (total + alignment - 1) // alignment * alignment
    if padded > I64_MAX:
        raise ValueError("padded extent overflow")
    return PLEGeometry(primes, offsets, factors, total, padded, vocab_size)
