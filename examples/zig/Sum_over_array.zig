export fn sumArray(array: [*]const u32, len: usize) u32 {
    var sum: u32 = 0;
    for (array[0..len]) |item| {
        sum += item;
    }
    return sum;
}
