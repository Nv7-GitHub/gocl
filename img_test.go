package gocl

import (
	"image"
	"image/color"
	"testing"
)

func TestInvert(t *testing.T) {
	var kernelSource = `
	__kernel void invert(
		__read_only image2d_t in,
		const int width,
		const int height,
		__write_only image2d_t out)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		float4 pixel = (float4)(0);
		if ((pos.x < width) && (pos.y < height)) {
			pixel = read_imagef(in, pos);
			pixel = (float4)(1) - pixel;
			write_imagef(out, pos, pixel);
		}
	}`

	// Created all-black image
	inImg := image.NewRGBA(image.Rect(0, 0, 7, 7))
	b := inImg.Bounds()

	outImg, err := ExecuteImage(kernelSource, "invert", 0, 0, inImg, int32(b.Dx()), int32(b.Dy()))
	if err != nil {
		t.Fatal(err)
	}

	//Output should be all-white image
	width := outImg.Bounds().Dx()
	height := outImg.Bounds().Dy()
	correct := color.White
	r1, g1, b1, a1 := correct.RGBA()
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			color := outImg.At(x, y)
			r2, g2, b2, a2 := color.RGBA()
			if r1 != r2 || g1 != g2 || b1 != b2 || a1 != a2 {
				t.Fatalf("Invert failed: Expected color was r: %d, g: %d, b: %d, a: %d, Got color r: %d, g: %d, b: %d, a: %d", r1, g1, b1, a1, r2, b2, g2, a2)
			}
		}
	}
}
