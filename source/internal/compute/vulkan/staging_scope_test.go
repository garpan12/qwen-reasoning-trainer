package vulkan_test

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
)

// A large download inside an allocation scope goes through a staging buffer;
// EndScope must not free that buffer a second time (it used to enter the pool
// twice and be handed to two later allocations).
func TestVulkanDownloadInScopeNoDoubleFree(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skip(err)
	}
	defer v.Close()
	const n = 300_000 // > 64 KiB: the staging path on mapped buffers
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i % 1000)
	}
	for round := 0; round < 50; round++ {
		v.BeginScope()
		a := up(t, v, []int{n}, vals)
		got := down(t, v, a)
		if got[n-1] != vals[n-1] || got[12345] != vals[12345] {
			t.Fatalf("round %d: bad download", round)
		}
		v.EndScope()
		// Two live buffers of the same size class must never alias.
		x := up(t, v, []int{n}, vals)
		y := up(t, v, []int{n}, make([]float32, n))
		if gx := down(t, v, x); gx[n-1] != vals[n-1] {
			t.Fatalf("round %d: buffers alias (x overwritten)", round)
		}
		v.Free(x)
		v.Free(y)
	}
}
