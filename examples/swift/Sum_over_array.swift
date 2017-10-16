// Imperative style sum over an array of integers
func imperativeSum(input: [Int]) -> Int {
    var sum = 0
    for value in input {
        sum += value
    }
    return sum
}

// Functional style sum over an array of integers
func functionalSum(input: [Int]) -> Int {
    return input.reduce(0, +)
}
