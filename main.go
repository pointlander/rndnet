// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
)

// Find lfsr
var LFSR = flag.Bool("lfsr", false, "find a lfsr")

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

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
	}

	lfsr := uint32(1)
	for i := 0; i < 10; i++ {
		lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
		fmt.Println(lfsr)
	}	
}
