function square(a) {
	let result = a * a;
	return result;
}

// Call function once to fill type information
square(23);

// Call function again to go from uninitialized -> pre-monomorphic -> monomorphic
square(13);
%OptimizeFunctionOnNextCall(square);
square(71);

