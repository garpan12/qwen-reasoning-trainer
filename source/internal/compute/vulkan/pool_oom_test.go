package vulkan

import (
	"os"
	"testing"
)

// TestVulkanAllocDrainsPoolOnOOM fills VRAM, frees everything into the pool,
// then asks for a buffer of a different size class. That allocation only
// succeeds if the backend drains the idle pool when vkAllocateMemory fails.
// It exhausts device memory, so it only runs with QWEN38_OOM_TEST=1 on an
// otherwise idle GPU.
func TestVulkanAllocDrainsPoolOnOOM(t *testing.T) {
	if os.Getenv("QWEN38_OOM_TEST") != "1" {
		t.Skip("set QWEN38_OOM_TEST=1 on an idle GPU (fills VRAM)")
	}
	b, err := New()
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()
	b.poolCapBytes = 1 << 40 // let the pool hold all of VRAM
	const chunk = 1 << 28    // 1 GiB of float32
	var held []*buffer
	for len(held) < 1024 {
		x, err := b.newF32Buffer(chunk)
		if err != nil {
			break
		}
		held = append(held, x)
	}
	if len(held) < 2 {
		t.Fatalf("only %d GiB allocatable", len(held))
	}
	for _, x := range held {
		b.Free(x)
	}
	t.Logf("filled %d GiB, pool now holds %.1f GiB", len(held), float64(b.poolBytes)/(1<<30))
	before := b.stats.PoolDrains
	y, err := b.newF32Buffer(chunk + chunk/2) // 1.5 GiB: a different size class
	if err != nil {
		t.Fatalf("allocation after filling the pool failed: %v", err)
	}
	defer b.Free(y)
	if b.stats.PoolDrains == before {
		t.Fatalf("expected a pool drain, got none")
	}
	t.Logf("1.5 GiB allocation succeeded after %d pool drain(s)", b.stats.PoolDrains-before)
}
