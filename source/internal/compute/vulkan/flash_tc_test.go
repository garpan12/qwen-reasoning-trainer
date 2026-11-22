package vulkan_test

import (
	"math"
	"os"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestFlashTCParity compares the tensor-core prefill attention
// (attention_flash_q4k_tc, QWEN38_FLASH_TC=1) with the fp32 flash kernel on
// the Qwen3.8 shape (24 q-heads, 4 kv-heads, headDim 256, Q4_K cache).
func TestFlashTCParity(t *testing.T) {
	vb, err := vulkan.New()
	if err != nil {
		t.Skip(err)
	}
	defer vb.Close()
	const hd, nHead, nHeadKV = 256, 24, 4
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for _, tc := range []struct{ nQ, nKV int }{{16, 16}, {64, 64}, {64, 700}, {128, 2100}, {192, 1000}, {2048, 6000}} {
		k := autograd.Random(401, 1.0, hd, nHeadKV, tc.nKV)
		v := autograd.Random(402, 1.0, hd, nHeadKV, tc.nKV)
		q := autograd.Random(403, 3.0, hd, nHead, tc.nQ)
		kraw, _ := quant.Quantize(quant.TypeQ4_K, k.F32)
		vraw, _ := quant.Quantize(quant.TypeQ4_K, v.F32)
		kc, _ := vb.AllocKV(hd, nHeadKV, tc.nKV, quant.TypeQ4_K)
		vc, _ := vb.AllocKV(hd, nHeadKV, tc.nKV, quant.TypeQ4_K)
		writeKVHeadMajor(t, vb, kc, kraw, tc.nKV, nHeadKV)
		writeKVHeadMajor(t, vb, vc, vraw, tc.nKV, nHeadKV)
		qb := up(t, vb, q.Dims, q.F32)
		run := func(env string) []float32 {
			os.Setenv("QWEN38_FLASH_TC", env)
			defer os.Unsetenv("QWEN38_FLASH_TC")
			out, err := vb.AttentionCached(qb, kc, vc, nHead, nHeadKV, tc.nKV, scale)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(out)
			return down(t, vb, out)
		}
		ref, got := run("0"), run("1")
		var dot, nr, ng, maxd float64
		for i := range ref {
			d := math.Abs(float64(got[i] - ref[i]))
			if d > maxd {
				maxd = d
			}
			dot += float64(got[i]) * float64(ref[i])
			nr += float64(ref[i]) * float64(ref[i])
			ng += float64(got[i]) * float64(got[i])
		}
		cos := dot / math.Sqrt(nr*ng)
		t.Logf("nQ=%d nKV=%d: max|diff|=%.5f cosine=%.7f", tc.nQ, tc.nKV, maxd, cos)
		if cos < 0.9999 || maxd > 0.02 {
			t.Errorf("nQ=%d nKV=%d: tensor-core attention diverges (max %.5f, cos %.7f)", tc.nQ, tc.nKV, maxd, cos)
		}
		vb.Free(qb)
		vb.Free(kc)
		vb.Free(vc)
	}
}
