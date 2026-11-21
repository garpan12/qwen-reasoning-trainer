package inference

import (
	"log"
	"os"
	"strconv"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// prefixEntry is the state of a finished request, rewound to the end of
// tokens: the prompt up to (not including) its last <|im_start|>, i.e. the
// conversation before the turn being generated. Agent clients resend the
// whole conversation each turn, so the next prompt usually starts with
// tokens and only its remainder needs prefilling.
type prefixEntry struct {
	st       *qwen38.State
	tokens   []int32
	capacity int
	reserved uint64
}

// prefixCacheEnabled reports whether requests reuse the previous state.
// QWEN38_PREFIX_CACHE=0 disables it.
func prefixCacheEnabled() bool { return os.Getenv("QWEN38_PREFIX_CACHE") != "0" }

// prefixSlots is how many request states the prefix cache keeps.
// QWEN38_PREFIX_SLOTS overrides the default of 4.
func prefixSlots() int {
	if n, err := strconv.Atoi(os.Getenv("QWEN38_PREFIX_SLOTS")); err == nil && n >= 1 {
		return n
	}
	return 4
}

// takePrefix removes and returns the cached entry whose tokens are the longest
// prefix of prompt not past limit, with at least capacity (nil if none), so the
// request owns it exclusively.
func (e *Engine) takePrefix(prompt []int32, limit, capacity int) *prefixEntry {
	e.prefixMu.Lock()
	defer e.prefixMu.Unlock()
	best := -1
	for i, c := range e.prefixes {
		if len(c.tokens) == 0 || len(c.tokens) > limit || c.capacity < capacity || !hasTokenPrefix(prompt, c.tokens) {
			continue
		}
		if best < 0 || len(c.tokens) > len(e.prefixes[best].tokens) {
			best = i
		}
	}
	if best < 0 {
		return nil
	}
	c := e.prefixes[best]
	e.prefixes = append(e.prefixes[:best], e.prefixes[best+1:]...)
	return c
}

// putPrefix caches c as the most recently used entry. Entries that c extends
// are redundant and dropped, as are the least recently used ones beyond
// prefixSlots.
func (e *Engine) putPrefix(c *prefixEntry) {
	e.prefixMu.Lock()
	keep := []*prefixEntry{c}
	var drop []*prefixEntry
	for _, o := range e.prefixes {
		if hasTokenPrefix(c.tokens, o.tokens) {
			drop = append(drop, o)
		} else {
			keep = append(keep, o)
		}
	}
	for len(keep) > prefixSlots() {
		drop = append(drop, keep[len(keep)-1])
		keep = keep[:len(keep)-1]
	}
	e.prefixes = keep
	e.prefixMu.Unlock()
	for _, o := range drop {
		e.dropPrefix(o)
	}
}

// evictOldestPrefix drops the least recently used cached entry; false if none.
func (e *Engine) evictOldestPrefix() bool {
	e.prefixMu.Lock()
	if len(e.prefixes) == 0 {
		e.prefixMu.Unlock()
		return false
	}
	c := e.prefixes[len(e.prefixes)-1]
	e.prefixes = e.prefixes[:len(e.prefixes)-1]
	e.prefixMu.Unlock()
	e.dropPrefix(c)
	return true
}

// dropAllPrefixes frees every cached entry.
func (e *Engine) dropAllPrefixes() {
	for e.evictOldestPrefix() {
	}
}

// cachedPrefixes reports how many entries are cached.
func (e *Engine) cachedPrefixes() int {
	e.prefixMu.Lock()
	defer e.prefixMu.Unlock()
	return len(e.prefixes)
}

// dropPrefix frees a cached state and its KV reservation.
func (e *Engine) dropPrefix(c *prefixEntry) {
	_ = compute.Run(e.Backend, func() error {
		c.st.Free()
		return nil
	})
	e.releaseKV(c.reserved)
}

// prefixSplit is where the reusable prefix of prompt ends: its last
// <|im_start|> (the header of the turn being generated). 0 means no split.
func (e *Engine) prefixSplit(prompt []int32) int {
	if !e.hasImStart {
		return 0
	}
	for i := len(prompt) - 1; i > 0; i-- {
		if prompt[i] == e.imStart {
			return i
		}
	}
	return 0
}

func hasTokenPrefix(s, prefix []int32) bool {
	if len(prefix) > len(s) {
		return false
	}
	for i, t := range prefix {
		if s[i] != t {
			return false
		}
	}
	return true
}

func logPrefix(reused, total, cached int) {
	log.Printf("prefix cache: reused %d of %d prompt tokens, prefilling %d (%d other cached)", reused, total, total-reused, cached)
}
