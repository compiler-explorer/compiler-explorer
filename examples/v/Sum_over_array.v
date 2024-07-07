import arrays

fn sum_array(array []int) !int {
	return arrays.reduce(array, fn (acc int, i int) int {
		return acc + i
	})
}

fn main() {
	a := [1, 2, 3, 4, 5]
	println(sum_array(a) or { 0 })
}
