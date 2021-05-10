package gocl

import (
	"math/rand"
	"testing"
)

func TestBufferInt(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global int * in,
	__global int * out) 
{
	out[0] = in[0] * in[0];
}
`
	prog, err := NewProgram(kernelSource, "square", 0, 0, DeviceTypeAll)
	if err != nil {
		t.Fatal(err)
	}

	input := int32(rand.Intn(1000))
	buff, err := prog.NewBufferFromInt(input)
	if err != nil {
		t.Fatal(err)
	}

	var empty int32
	out, err := prog.NewBufferFromInt(empty)
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{1}, []int{1}, buff, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataInt()
	if err != nil {
		t.Fatal(err)
	}

	if results != input*input {
		t.Fatalf("expected %d, not %d for square of %d", input*input, results, input)
	}
}

func TestBufferFloat(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global float * in,
	__global float * out) 
{
	out[0] = in[0] * in[0];
}
`
	prog, err := NewProgram(kernelSource, "square", 0, 0, DeviceTypeAll)
	if err != nil {
		t.Fatal(err)
	}

	input := float32(rand.Intn(1000))
	buff, err := prog.NewBufferFromFloat(input)
	if err != nil {
		t.Fatal(err)
	}

	var empty float32
	out, err := prog.NewBufferFromFloat(empty)
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{1}, []int{1}, buff, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataFloat()
	if err != nil {
		t.Fatal(err)
	}

	if results != input*input {
		t.Fatalf("expected %f, not %f for square of %f", input*input, results, input)
	}
}
