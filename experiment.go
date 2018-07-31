// Verify that importance sampling is a valid way to
// compute the probability that a model will produce a
// sequence that contains a subset of its alphabet.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/approb"
)

const (
	SeqLen     = 5
	NumSamples = 10000000
)

func main() {
	rand.Seed(time.Now().UnixNano())

	expectedMean, expectedVar := approb.Moments(NumSamples, func() float64 {
		val := true
		for _, num := range sampleSequence() {
			if num == 0 {
				val = false
			}
		}
		if val {
			return 1.0
		} else {
			return 0.0
		}
	})
	fmt.Println("expected prob:", expectedMean, "+/-", math.Sqrt(expectedVar/NumSamples))

	actualMean, actualVar := approb.Moments(NumSamples, func() float64 {
		return importanceSample()
	})
	fmt.Println("actual prob:", actualMean, "+/-", math.Sqrt(actualVar/NumSamples))
}

func sampleSequence() []int {
	var last int
	var res []int
	for i := 0; i < SeqLen; i++ {
		last = sampleElement(probDist(last))
		res = append(res, last)
	}
	return res
}

func importanceSample() float64 {
	var last int
	prob := 1.0
	for i := 0; i < SeqLen; i++ {
		dist := probDist(last)
		prob *= (1 - dist[0])
		dist[0] = 0
		sum := dist[1] + dist[2]
		dist[1] /= sum
		dist[2] /= sum
		last = sampleElement(dist)
	}
	return prob
}

func probDist(last int) [3]float64 {
	if last == 0 {
		return [3]float64{0.3, 0.5, 0.2}
	} else if last == 1 {
		return [3]float64{0.1, 0.2, 0.7}
	} else {
		return [3]float64{0.5, 0.3, 0.2}
	}
}

func sampleElement(dist [3]float64) int {
	num := rand.Float64()
	if num < dist[0] {
		return 0
	} else if num < dist[0]+dist[1] {
		return 1
	} else {
		return 2
	}
}
