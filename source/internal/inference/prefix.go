package inference

import (
	"log"
	"os"

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

// takePrefix removes and returns the cached entry (nil if none), so a request
// owns it exclusively.
func (e *Engine) takePrefix() *prefixEntry {
	e.prefixMu.Lock()
	defer e.prefixMu.Unlock()
	c := e.prefix
	e.prefix = nil
	return c
}

// putPrefix caches c, dropping any entry another request stored meanwhile.
func (e *Engine) putPrefix(c *prefixEntry) {
	e.prefixMu.Lock()
	old := e.prefix
	e.prefix = c
	e.prefixMu.Unlock()
	if old != nil {
		e.dropPrefix(old)
	}
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

func logPrefix(reused, total int) {
	log.Printf("prefix cache: reused %d of %d prompt tokens, prefilling %d", reused, total, total-reused)
}
