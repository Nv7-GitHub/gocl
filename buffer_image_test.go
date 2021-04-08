package gocl

import (
	"image"
	"image/png"
	"os"
	"testing"
)

func TestBufferImage(t *testing.T) {
	file, err := os.Open("gopher.png")
	if err != nil {
		panic(err)
	}
	img, err := png.Decode(file)
	if err != nil {
		panic(err)
	}
	file.Close()

	var kernelSource = `
	__kernel void invert(
		__read_only image2d_t in,
		__write_only image2d_t out)
	{
		const int2 pos = (int2)(get_global_id(0), get_global_id(1));
		const int2 dim = get_image_dim(in);
		float4 pixel = (float4)(0);
		if (pos.x < dim.x && pos.y < dim.y) {
			pixel = read_imagef(in, pos);
			pixel = (float4)(1) - pixel;
			pixel.w = 1;
			write_imagef(out, pos, pixel);
		}
	}`

	prog, err := NewProgram(kernelSource, "invert", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	input, err := prog.NewBufferFromImage(img)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Cleanup()

	out, err := prog.NewBufferFromImage(image.NewRGBA(img.Bounds()))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Cleanup()

	err = prog.Execute([]int{img.Bounds().Dx(), img.Bounds().Dy()}, input, out)
	if err != nil {
		t.Fatal(err)
	}

	output, err := out.GetDataImage()
	if err != nil {
		t.Fatal(err)
	}

	outFile, err := os.OpenFile("out.png", os.O_CREATE|os.O_WRONLY, os.ModePerm)
	if err != nil {
		t.Fatal(err)
	}

	err = png.Encode(outFile, output)
	if err != nil {
		t.Fatal(err)
	}
}
