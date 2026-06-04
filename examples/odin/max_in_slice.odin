package main

@(export)
max_slice :: proc(array: []int) -> int {
	max_val := min(int)
	for elem in array {
		max_val = max(max_val, elem)
	}
	return max_val
}
