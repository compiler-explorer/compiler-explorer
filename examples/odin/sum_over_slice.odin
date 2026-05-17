package main

@(export)
sum_slice :: proc(array: []int) -> int {
	sum := 0
	for elem in array {
		sum += elem
	}
	return sum
}
