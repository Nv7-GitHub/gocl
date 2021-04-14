package gocl

import (
	"math/rand"
	"testing"
)

func TestFloatArr(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global float * in,
	const unsigned int count,
	__global float * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]float32, 1024)
	for i := 0; i < len(input); i++ {
		input[i] = rand.Float32()
	}

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromFloatArr(input)
	if err != nil {
		t.Fatal(err)
	}

	out, err := prog.NewEmptyFloat32Arr(len(input))
	if err != nil {
		t.Fatal(err)
	}

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataFloatArr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %f, Got: %f", val*val, results[i])
		}
	}
}

func BenchmarkFloatArr(t *testing.B) {
	var kernelSource = `
__kernel void square(
	__global float * in,
	const unsigned int count,
	__global float * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]float32, t.N)
	for i := 0; i < len(input); i++ {
		input[i] = rand.Float32()
	}

	t.ResetTimer()

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromFloatArr(input)
	if err != nil {
		t.Fatal(err)
	}
	defer buff.Cleanup()

	out, err := prog.NewEmptyFloat32Arr(len(input))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Cleanup()

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataFloatArr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %f, Got: %f", val*val, results[i])
		}
	}
}

func TestIntArr(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global int * in,
	const unsigned int count,
	__global int * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]int32, 1024)
	for i := 0; i < len(input); i++ {
		input[i] = int32(rand.Int())
	}

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromIntArr(input)
	if err != nil {
		t.Fatal(err)
	}

	empty := make([]int32, len(input))
	out, err := prog.NewBufferFromIntArr(empty)
	if err != nil {
		t.Fatal(err)
	}

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataIntArr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %d, Got: %d", val*val, results[i])
		}
	}
}

func BenchmarkIntArr(t *testing.B) {
	var kernelSource = `
__kernel void square(
	__global int * in,
	const unsigned int count,
	__global int * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]int32, t.N)
	for i := 0; i < len(input); i++ {
		input[i] = int32(rand.Int())
	}

	t.ResetTimer()

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromIntArr(input)
	if err != nil {
		t.Fatal(err)
	}
	defer buff.Cleanup()

	empty := make([]int32, len(input))
	out, err := prog.NewBufferFromIntArr(empty)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Cleanup()

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataIntArr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %d, Got: %d", val*val, results[i])
		}
	}
}
