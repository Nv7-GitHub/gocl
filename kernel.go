package gocl

import (
	"math"

	"github.com/Nv7-Github/go-cl"
)

func NewProgram(src string, kernelName string, platform, device int) (*Program, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	p := platforms[platform]

	devices, err := p.GetDevices(cl.DeviceTypeAll)
	if err != nil {
		return nil, err
	}

	d := devices[device]

	ctx, err := cl.CreateContext([]*cl.Device{d})
	if err != nil {
		return nil, err
	}

	prog, err := ctx.CreateProgramWithSource([]string{src})
	if err != nil {
		return nil, err
	}

	err = prog.BuildProgram(nil, "")
	if err != nil {
		return nil, err
	}

	queue, err := ctx.CreateCommandQueue(d, 0)
	if err != nil {
		return nil, err
	}
	kernel, err := prog.CreateKernel(kernelName)
	if err != nil {
		return nil, err
	}

	return &Program{
		prog:       prog,
		ctx:        ctx,
		kernelName: kernelName,
		queue:      queue,
		kernel:     kernel,
		args:       make([]interface{}, 0),
		d:          d,
	}, nil
}

type Program struct {
	prog       *cl.Program
	ctx        *cl.Context
	kernelName string
	args       []interface{}

	kernel *cl.Kernel
	queue  *cl.CommandQueue
	d      *cl.Device
}

// Execute executes the kernel. The length should be the [width, height] for an image and [length] for an array.
func (p *Program) Execute(length []int, args ...Argument) error {
	ars := make([]interface{}, len(args))
	for i, val := range args {
		ars[i] = val.val()
	}
	err := p.kernel.SetArgs(ars...)
	if err != nil {
		return err
	}

	local, err := p.kernel.PreferredWorkGroupSizeMultiple(p.d)
	if err != nil {
		return err
	}
	local = int(math.Floor(math.Sqrt(float64(local))))

	_, err = p.queue.EnqueueNDRangeKernel(p.kernel, nil, length, []int{local, local}, nil)
	if err != nil {
		return err
	}

	err = p.queue.Finish()
	return err
}
