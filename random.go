// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/pointlander/datum/iris"
)

// RandomLayer is a random neural network layer
type RandomLayer struct {
	Rows    int
	Columns int
	Rand    Rand
}

// RandomNetwork is a random neural network
type RandomNetwork []RandomLayer

// Inference performs inference on a neural network
func (n RandomNetwork) Inference(inputs, outputs []float32) {
	last := len(n) - 1
	for i, layer := range n {
		rnd := layer.Rand
		columns := len(outputs)
		if i < len(n)-1 {
			columns = n[i+1].Columns
		}
		values, factor :=
			make([]float32, columns),
			float32(math.Sqrt(2/float64(columns)))
		for j := 0; j < layer.Rows; j++ {
			sum := (2*rnd.Float32() - 1) * factor
			for _, input := range inputs {
				sum += input * (2*rnd.Float32() - 1) * factor
			}
			e := float32(math.Exp(float64(sum)))
			values[j] = e / (e + 1)
		}
		if i == last {
			copy(outputs, values)
		} else {
			inputs = values
		}
	}
}

// Copy copies a network
func (n RandomNetwork) Copy() RandomNetwork {
	var network RandomNetwork
	for _, layer := range n {
		l := RandomLayer{
			Rows:    layer.Rows,
			Columns: layer.Columns,
			Rand:    layer.Rand,
		}
		network = append(network, l)
	}
	return network
}

// RandomNetworkModel is the real network model
func RandomNetworkModel(seed int) float64 {
	rnd := Rand(LFSRInit + seed)
	type Genome struct {
		Network RandomNetwork
		Fitness float32
	}
	var genomes []Genome
	addNetwork := func(i int) {
		var network RandomNetwork
		layer := RandomLayer{
			Rows:    4,
			Columns: 4,
			Rand:    Rand(LFSRInit + i + seed + NumGenomes),
		}
		network = append(network, layer)

		layer = RandomLayer{
			Rows:    3,
			Columns: 4,
			Rand:    Rand(LFSRInit + i + seed + 2*NumGenomes),
		}
		network = append(network, layer)

		genomes = append(genomes, Genome{
			Network: network,
		})
	}
	for i := 0; i < NumGenomes; i++ {
		addNetwork(i)
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	inputs, outputs := make([]float32, 4), make([]float32, 3)
	get := func() int {
		for {
			for i, genome := range genomes {
				if rnd.Float32() > genome.Fitness {
					return i
				}
			}
		}
	}
	i := 0
	for {
		for j, genome := range genomes {
			sum := float32(0)
			for _, flower := range datum.Fisher {
				for k, value := range flower.Measures {
					inputs[k] = float32(value)
				}
				genome.Network.Inference(inputs, outputs)
				expected := make([]float32, 3)
				expected[iris.Labels[flower.Label]] = 1
				loss := float32(0)
				for l, output := range outputs {
					diff := expected[l] - output
					loss += diff * diff
				}
				loss = float32(math.Sqrt(float64(loss)))
				sum += loss
			}
			sum /= float32(len(datum.Fisher)) * float32(math.Sqrt(3))
			genomes[j].Fitness = sum
		}
		sort.Slice(genomes, func(i, j int) bool {
			return genomes[i].Fitness < genomes[j].Fitness
		})
		genomes = genomes[:NumGenomes]
		i++
		if i > 127 {
			break
		}

		for i := 0; i < 256; i++ {
			a, b := get(), get()
			layer := rnd.Uint32() & 1
			networkA, networkB :=
				genomes[a].Network.Copy(), genomes[b].Network.Copy()
			layerA, layerB := networkA[layer], networkB[layer]
			layerA.Rand = layerA.Rand ^ layerB.Rand
			genomes = append(genomes, Genome{
				Network: networkA,
			})
		}
	}

	network := genomes[0].Network
	misses, total := 0, 0
	for _, flower := range datum.Fisher {
		for k, value := range flower.Measures {
			inputs[k] = float32(value)
		}
		network.Inference(inputs, outputs)
		max, index := float32(0), 0
		for j, output := range outputs {
			if output > max {
				max, index = output, j
			}
		}
		if index != iris.Labels[flower.Label] {
			misses++
		}
		total++
	}
	quality := float64(misses) / float64(total)
	fmt.Println(genomes[0].Fitness, quality)
	return quality
}
