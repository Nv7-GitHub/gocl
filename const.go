package gocl

import (
	"errors"
)

type Const struct {
	dat interface{}
}

func getConstData(data interface{}) (interface{}, error) {
	switch val := data.(type) {
	default:
		return nil, errors.New("gocl: constant must be int, float, or uint")
	case int32:
		return val, nil
	case uint32:
		return val, nil
	case float32:
		return val, nil
	case int:
		return int32(val), nil
	case int64:
		return int32(val), nil
	case uint:
		return uint32(val), nil
	case uint16:
		return uint32(val), nil
	case uint8:
		return uint32(val), nil
	case uint64:
		return uint32(val), nil
	case float64:
		return float32(val), nil
	}
}

func (c *Const) val() interface{} {
	return c.dat
}

func (c *Const) SetData(data interface{}) error {
	data, err := getConstData(data)
	if err != nil {
		return err
	}
	c.dat = data
	return nil
}

func NewConst(data interface{}) (*Const, error) {
	data, err := getConstData(data)
	if err != nil {
		return nil, err
	}
	return &Const{
		dat: data,
	}, nil
}
