import sys
from collections import OrderedDict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CacheConfig:
    """Holds the parsed cache configuration parameters."""

    cache_line_size: int   # bytes per cache line (block)
    cache_inclusive: bool  # True if L2 is inclusive of L1
    l1_num_ways: int       # associativity of L1
    l1_data_size: int      # total data size of L1 in bytes
    l2_num_ways: int       # associativity of L2
    l2_data_size: int      # total data size of L2 in bytes

    # Derived values (number of sets per level).
    l1_num_sets: int = 0
    l2_num_sets: int = 0


def num_sets(data_size: int, cache_line_size: int, num_ways: int) -> int:
    """Calculate the number of sets in a cache.

    NUM_SETS = DATA_SIZE / (CACHE_LINE_SIZE * NUM_WAYS)
    """
    return data_size // (cache_line_size * num_ways)


def parse_config(config_file: str) -> CacheConfig:
    """Parse the config file.

    The file contains a single line of comma-separated values:
        CACHE_LINE_SIZE, CACHE_INCLUSIVE, L1_NUM_WAYS, L1_DATA_SIZE,
        L2_NUM_WAYS, L2_DATA_SIZE

    CACHE_INCLUSIVE is a boolean ('TRUE' or 'FALSE'); all numbers are
    powers of 2.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Expected 6 comma-separated values in config, got {len(parts)}: {line!r}"
        )

    cache_line_size = int(parts[0])
    cache_inclusive = parts[1].strip().upper() == "TRUE"
    l1_num_ways = int(parts[2])
    l1_data_size = int(parts[3])
    l2_num_ways = int(parts[4])
    l2_data_size = int(parts[5])

    config = CacheConfig(
        cache_line_size=cache_line_size,
        cache_inclusive=cache_inclusive,
        l1_num_ways=l1_num_ways,
        l1_data_size=l1_data_size,
        l2_num_ways=l2_num_ways,
        l2_data_size=l2_data_size,
    )

    config.l1_num_sets = num_sets(l1_data_size, cache_line_size, l1_num_ways)
    config.l2_num_sets = num_sets(l2_data_size, cache_line_size, l2_num_ways)

    return config


# ---------------------------------------------------------------------------
# Address decoding
# ---------------------------------------------------------------------------
@dataclass
class AddressFields:
    """The decomposed parts of a memory address for a given cache level."""

    tag: int
    set_index: int
    block_offset: int


def decode_address(
    address: int, cache_line_size: int, num_sets_value: int
) -> AddressFields:
    """Split a 32-bit address into Tag, Set Index, and Block Offset.

    Layout (high -> low bits):
        | Tag | Set Index | Block Offset |

    All sizes are powers of 2, so the number of bits is log2 of the size.
    """
    offset_bits = (cache_line_size - 1).bit_length()
    index_bits = (num_sets_value - 1).bit_length() if num_sets_value > 1 else 0

    block_offset = address & (cache_line_size - 1)
    if num_sets_value > 1:
        set_index = (address >> offset_bits) & (num_sets_value - 1)
    else:
        set_index = 0
    tag = address >> (offset_bits + index_bits)

    return AddressFields(tag=tag, set_index=set_index, block_offset=block_offset)


def parse_address(token: str) -> int:
    """Parse a 32-bit hex address token (with or without a '0x' prefix)."""
    return int(token, 16)


# ---------------------------------------------------------------------------
# Cache (single level)
# ---------------------------------------------------------------------------
@dataclass
class CacheLine:
    """A single cache line. `dirty` is tracked for future write-back logic."""

    tag: int
    dirty: bool = False


class Cache:
    """Represents a single cache level (usable for both L1 and L2).

    The cache is an array of sets. Each set holds up to ``num_ways`` lines
    and is implemented as an ``OrderedDict`` keyed by tag, which gives us an
    O(1) LRU ordering:

        * The MRU (most-recently-used) line is at the end of the dict.
        * The LRU (least-recently-used) line is at the front and is the
          eviction candidate when the set is full.
    """

    def __init__(
        self,
        cache_line_size: int,
        num_ways: int,
        data_size: int,
    ) -> None:
        self.cache_line_size = cache_line_size
        self.num_ways = num_ways
        self.data_size = data_size
        self.num_sets = num_sets(data_size, cache_line_size, num_ways)

        # One OrderedDict[tag -> CacheLine] per set.
        self.sets: list[OrderedDict[int, CacheLine]] = [
            OrderedDict() for _ in range(self.num_sets)
        ]

    # -- internal helpers ---------------------------------------------------
    def _decode(self, address: int) -> AddressFields:
        return decode_address(address, self.cache_line_size, self.num_sets)

    def _touch(self, cache_set: "OrderedDict[int, CacheLine]", tag: int) -> None:
        """Mark ``tag`` as most-recently-used within its set."""
        cache_set.move_to_end(tag)

    def _insert(
        self, cache_set: "OrderedDict[int, CacheLine]", tag: int, dirty: bool
    ) -> int | None:
        """Insert a new line, evicting the LRU line if the set is full.

        Returns the evicted tag (or ``None`` if no eviction occurred).
        """
        evicted_tag: int | None = None
        if len(cache_set) >= self.num_ways:
            # popitem(last=False) removes the LRU (front) entry.
            evicted_tag, _ = cache_set.popitem(last=False)

        cache_set[tag] = CacheLine(tag=tag, dirty=dirty)
        # New line is inserted at the end (MRU) by default.
        return evicted_tag

    # -- public API ---------------------------------------------------------
    def probe(self, address: int) -> CacheLine | None:
        """Look up an address. Returns the line on hit (and updates its LRU
        status), or ``None`` on miss. Performs no allocation.

        Because the LRU status is updated whenever a line is found, this is
        safe to use for both read and write hit handling.
        """
        fields = self._decode(address)
        cache_set = self.sets[fields.set_index]

        line = cache_set.get(fields.tag)
        if line is not None:
            self._touch(cache_set, fields.tag)  # LRU update on hit (read or write)
        return line

    def allocate(self, address: int, dirty: bool = False) -> int | None:
        """Allocate a line for ``address``, evicting the LRU line if the set
        is full. Returns the evicted tag (or ``None``)."""
        fields = self._decode(address)
        cache_set = self.sets[fields.set_index]
        return self._insert(cache_set, fields.tag, dirty=dirty)

    def read(self, address: int) -> bool:
        """Process a read access. Returns True on hit, False on miss.

        On a hit, the line's LRU status is updated. On a miss, the line is
        allocated (and the LRU line evicted if necessary).
        """
        if self.probe(address) is not None:
            return True

        # Miss: allocate the line (clean, since it was just read in).
        self.allocate(address, dirty=False)
        return False

    def write(self, address: int) -> bool:
        """Process a write access. Returns True on hit, False on miss.

        On both hit and miss the LRU status is updated; the line is marked
        dirty.
        """
        line = self.probe(address)
        if line is not None:
            line.dirty = True  # LRU already updated by probe()
            return True

        # Write miss: allocate the line as dirty (write-allocate policy).
        self.allocate(address, dirty=True)
        return False


# ---------------------------------------------------------------------------
# Cache hierarchy (L1 + L2)
# ---------------------------------------------------------------------------
class CacheHierarchy:
    """A two-level cache hierarchy wiring an L1 and L2 :class:`Cache` together.

    Access results are reported as one of three strings:
        * ``'L1HIT'``  - serviced by L1
        * ``'L2HIT'``  - missed in L1 but found in L2
        * ``'MEMACC'`` - missed in both levels (went to main memory)
    """

    L1HIT = "L1HIT"
    L2HIT = "L2HIT"
    MEMACC = "MEMACC"

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.l1 = Cache(
            cache_line_size=config.cache_line_size,
            num_ways=config.l1_num_ways,
            data_size=config.l1_data_size,
        )
        self.l2 = Cache(
            cache_line_size=config.cache_line_size,
            num_ways=config.l2_num_ways,
            data_size=config.l2_data_size,
        )

    def read(self, address: int) -> str:
        """Read access policy.

        * Hit in L1            -> 'L1HIT'.
        * Miss L1, hit L2      -> 'L2HIT', load the line into L1.
        * Miss both            -> 'MEMACC', load the line into BOTH L1 and L2.
        """
        if self.l1.probe(address) is not None:
            return self.L1HIT

        if self.l2.probe(address) is not None:
            # Promote into L1 (write-back/clean copy still resides in L2).
            self.l1.allocate(address, dirty=False)
            return self.L2HIT

        # Miss in both levels: fetch from memory into both L2 and L1.
        self.l2.allocate(address, dirty=False)
        self.l1.allocate(address, dirty=False)
        return self.MEMACC

    def write(self, address: int) -> str:
        """Write access policy: write-through, no-write-allocate.

        * Hit in L1            -> 'L1HIT'.
        * Miss L1, hit L2      -> 'L2HIT' (do NOT load into L1).
        * Miss both            -> 'MEMACC' (do NOT load into any level).
        """
        if self.l1.probe(address) is not None:
            return self.L1HIT

        if self.l2.probe(address) is not None:
            # No-write-allocate: leave L1 untouched.
            return self.L2HIT

        # No-write-allocate at all levels: nothing is loaded.
        return self.MEMACC


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    if len(argv) != 4:
        prog = argv[0] if argv else "cache_sim.py"
        print(
            f"Usage: {prog} <config_file> <trace_file> <output_file>",
            file=sys.stderr,
        )
        return 1

    config_file, trace_file, output_file = argv[1], argv[2], argv[3]

    config = parse_config(config_file)
    hierarchy = CacheHierarchy(config)

    results: list[str] = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines

            # Each line: "<ADDRESS> , W/R" (ADDRESS is hex, no '0x').
            addr_token, op_token = line.split(",")
            address = parse_address(addr_token.strip())
            op = op_token.strip().upper()

            if op == "W":
                result = hierarchy.write(address)
            else:  # "R"
                result = hierarchy.read(address)

            results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

