package gocl

import (
	"github.com/Nv7-Github/go-cl"
)

func NewProgram(src string, kernelName string, platform, device int, kind cl.DeviceType) (*Program, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	p := platforms[platform]

	devices, err := p.GetDevices(cl.DeviceType(kind))
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
func (p *Program) Execute(length []int, local []int, args ...Argument) error {
	ars := make([]interface{}, len(args))
	for i, val := range args {
		ars[i] = val.val()
	}
	err := p.kernel.SetArgs(ars...)
	if err != nil {
		return err
	}

	_, err = p.queue.EnqueueNDRangeKernel(p.kernel, nil, length, local, nil)
	if err != nil {
		return err
	}

	err = p.queue.Finish()
	return err
}

// PrefferedWorkGroupSize gets the preferred OpenCL local work group size
func (p *Program) PreferredWorkGroupSize() (int, error) {
	size, err := p.kernel.PreferredWorkGroupSizeMultiple(p.d)
	return size, err
}

// MaxWorkGroupSize gets the maximum OpenCL work group size for the kernel
func (p *Program) MaxWorkGroupSize() (int, error) {
	size, err := p.kernel.WorkGroupSize(p.d)
	return size, err
}

// DeviceWorkGroupSize gets the device's maximum OpenCL work group size for the kernel
func (p *Program) DeviceWorkGroupSize() int {
	return p.d.MaxWorkGroupSize()
}
