package labels

var N int

func Closures() {
	if N != 1 {
		go func() {
			myFunc := func() {
				if N != 4 {
					print(1)
				}
			}

			if N != 3 {
				myFunc()
			}
		}()
	}
}

func Closures_func1_1() {
	if N != 1 {
		print(1)
	}
}

func αβ() {
	if N != 1 {
		print(1)
	}
}
