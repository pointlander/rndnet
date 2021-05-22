// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
)

var (
	// LFSR find lfsr
	LFSR = flag.Bool("lfsr", false, "find a lfsr")
	// Real uses the real network
	Real = flag.Bool("real", false, "real network")
	// Shared uses the real network with shared weights
	Shared = flag.Bool("shared", false, "real network with share weights")
	// Complex uses the complex network
	Complex = flag.Bool("complex", false, "complex network")
)

const (
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
	// LFSRInit is an initial LFSR state
	LFSRInit = 0x55555555
	// NumGenomes is the number of genomes
	NumGenomes = 256
)

// Rand is a random number generator
type Rand uint32

// Float32 returns a random float32 between 0 and 1
func (r *Rand) Float32() float32 {
	lfsr := *r
	if lfsr&1 == 1 {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return float32(lfsr) / ((1 << 32) - 1)
}

// Uint32 returns a random uint32
func (r *Rand) Uint32() uint32 {
	lfsr := *r
	if lfsr&1 == 1 {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return uint32(lfsr)
}

func main() {
	flag.Parse()

	if *LFSR {
		// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
		// https://users.ece.cmu.edu/~koopman/lfsr/index.html
		count, polynomial := 0, uint32(0x80000000)
		for polynomial != 0 {
			lfsr, period := uint32(1), uint32(0)
			for {
				lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & polynomial)
				period++
				if lfsr == 1 {
					break
				}
			}
			fmt.Printf("%v period=%v\n", count, period)
			if period == math.MaxUint32 {
				fmt.Printf("%x\n", polynomial)
				return
			}
			count++
			polynomial++
		}
		return
	} else if *Real {
		RealNetworkModel()
		return
	} else if *Complex {
		ComplexNetworkModel()
		return
	} else if *Shared {
		SharedNetworkModel()
	}
}
