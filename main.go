// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"runtime"
)

var (
	// LFSR find lfsr
	LFSR = flag.Bool("lfsr", false, "find a lfsr")
	// Real uses the real network
	Real = flag.Bool("real", false, "real network")
	// Random is a random neural network
	Random = flag.Bool("random", false, "randome network")
	// Shared uses the real network with shared weights
	Shared = flag.Bool("shared", false, "real network with share weights")
	// Complex uses the complex network
	Complex = flag.Bool("complex", false, "complex network")
	// RNN uses the recurrent neural network
	RNN = flag.Bool("rnn", false, "recurrent neural network")
	// Search search for the best seed
	Search = flag.Bool("search", false, "search for the best seed")
)

const (
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
	// LFSRInit is an initial LFSR state
	LFSRInit = 0x55555555
	// NumGenomes is the number of genomes
	NumGenomes = 256
	// SearchIterations is the number of search iterations
	SearchIterations = 256
	// Size is the size of the recurrent neural network
	Size = 8
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

	type Result struct {
		Seed    int
		Quality float64
	}
	process := func(model func(seed int) float64) {
		results := make(chan Result, runtime.NumCPU())
		routine := func(seed int) {
			results <- Result{
				Seed:    seed,
				Quality: model(seed * NumGenomes),
			}
		}
		min, seed, count, j, flight := 1.0, 0, 0, 0, 0
		for i := 0; i < runtime.NumCPU() && j < SearchIterations; i++ {
			go routine(j)
			j++
			flight++
		}
		for j < SearchIterations {
			result := <-results
			flight--
			if result.Quality < min {
				min, seed = result.Quality, result.Seed
			}
			if result.Quality < .1 {
				count++
			}

			go routine(j)
			j++
			flight++
		}
		for i := 0; i < flight; i++ {
			result := <-results
			if result.Quality < min {
				min, seed = result.Quality, result.Seed
			}
			if result.Quality < .1 {
				count++
			}
		}
		fmt.Println(min, seed, count)
	}

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
		if *Search {
			process(RealNetworkModel)
		} else {
			// 0.02 135 14
			RealNetworkModel(135 * NumGenomes)
		}
		return
	} else if *Random {
		if *Search {
			process(RandomNetworkModel)
		} else {
			// 0.04666666666666667 1391 32
			RandomNetworkModel(1391 * NumGenomes)
		}
		return
	} else if *Complex {
		if *Search {
			process(ComplexNetworkModel)
		} else {
			// 0.05333333333333334 186 1
			ComplexNetworkModel(186 * NumGenomes)
		}
		return
	} else if *Shared {
		if *Search {
			// 0.06 152 1
			process(SharedNetworkModel)
		} else {
			SharedNetworkModel(152 * NumGenomes)
		}
		return
	} else if *RNN {
		activations, next, connections, factor :=
			make([]float32, Size), make([]float32, Size), make([][]float32, Size), float32(math.Sqrt(2/float64(Size)))
		for i := range connections {
			connections[i] = make([]float32, Size)
		}
		var average, standardDeviation float32
		for i := 0; i < 1024; i++ {
			rnd := Rand(LFSRInit + 3*Size*Size)
			for j := 0; j < Size; j++ {
				sum := (2*rnd.Float32() - 1) * factor
				for k := 0; k < Size; k++ {
					if weight := rnd.Float32(); k == j {
						sum += (2*weight - 1) * factor * activations[k]
					} else if (connections[j][k] < average-2*standardDeviation ||
						connections[j][k] > average+2*standardDeviation) && i > 128 {
						sum += (2*weight - 1) * factor * activations[k]
					}
				}
				e := float32(math.Exp(float64(sum)))
				next[j] = e / (e + 1)
			}
			for j, a := range next {
				for k, b := range next {
					if j == k {
						continue
					}
					if a > .5 && b > .5 {
						connections[j][k]++
					}
				}
			}
			copy(activations, next)
			sum, sumSquared := float32(0), float32(0)
			for _, a := range connections {
				for _, b := range a {
					sum += b
					sumSquared += b * b
				}
			}
			average = sum / (Size * Size)
			standardDeviation = float32(math.Sqrt(float64(sumSquared/(Size*Size) - average*average)))
			fmt.Println(average, standardDeviation, connections)
		}
	}
}
