function square(a) {
	let result = a * a;
	return result;
}

// Allocate feedback vector for the function and call it once to record type
// information.
%PrepareFunctionForOptimization(square)
square(23);

// Optimize with Turbofan.
%OptimizeFunctionOnNextCall(square);
square(71);
