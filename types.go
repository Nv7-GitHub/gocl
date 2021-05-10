package gocl

import "github.com/Nv7-Github/go-cl"

type Argument interface {
	val() interface{}
}

// For convenience
const (
	DeviceTypeCPU         = cl.DeviceTypeCPU
	DeviceTypeGPU         = cl.DeviceTypeGPU
	DeviceTypeAccelerator = cl.DeviceTypeAccelerator
	DeviceTypeDefault     = cl.DeviceTypeDefault
	DeviceTypeAll         = cl.DeviceTypeAll
)
