// Imperative style array of maximum values per index
func imperativeMaxArray(x: [Int], y: [Int]) -> [Int] {
    var maxima: [Int] = []
    let count = min(x.count, y.count)
    for index in 0..<count {
        maxima.append(max(x[index], y[index]))
    }
    return maxima
}

// Functional style array of maximum values per index
func functionalMaxArray(x: [Int], y: [Int]) -> [Int] {
    return zip(x, y).map(max)
}
