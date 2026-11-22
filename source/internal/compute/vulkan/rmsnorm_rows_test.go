package vulkan_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
)

// TestVulkanRMSNormAddRows checks the fused residual add + RMSNorm (incl. the
// wide decode variant used for <= 4 rows) against a host computation.
func TestVulkanRMSNormAddRows(t *testing.T) {
	vb, err := vulkan.New()
	if err != nil {
		t.Skip(err)
	}
	defer vb.Close()
	const eps = 1e-6
	for _, tc := range []struct{ rowLen, rows int }{{5120, 1}, {5120, 3}, {5120, 7}, {1000, 1}} {
		a := autograd.Random(501, 1.0, tc.rowLen, tc.rows)
		r := autograd.Random(502, 1.0, tc.rowLen, tc.rows)
		w := autograd.Random(503, 1.0, tc.rowLen)
		sum, norm, err := vb.RMSNormAdd(up(t, vb, a.Dims, a.F32), up(t, vb, r.Dims, r.F32), up(t, vb, w.Dims, w.F32), eps)
		if err != nil {
			t.Fatal(err)
		}
		gs, gn := down(t, vb, sum), down(t, vb, norm)
		for row := 0; row < tc.rows; row++ {
			var ss float64
			for i := 0; i < tc.rowLen; i++ {
				v := float64(a.F32[row*tc.rowLen+i] + r.F32[row*tc.rowLen+i])
				ss += v * v
			}
			inv := 1 / math.Sqrt(ss/float64(tc.rowLen)+eps)
			for i := 0; i < tc.rowLen; i++ {
				k := row*tc.rowLen + i
				s := float64(a.F32[k] + r.F32[k])
				want := s * inv * float64(w.F32[i])
				if math.Abs(float64(gs[k])-s) > 1e-5 || math.Abs(float64(gn[k])-want) > 1e-4+1e-4*math.Abs(want) {
					t.Fatalf("rowLen=%d rows=%d [%d]: sum %v want %v, norm %v want %v", tc.rowLen, tc.rows, k, gs[k], s, gn[k], want)
				}
			}
		}
	}
}
