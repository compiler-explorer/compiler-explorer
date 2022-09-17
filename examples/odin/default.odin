package foo

foreign {
	elsewhere :: proc "odin" () -> int ---
}

@(require) // to force the function to appear in the assembly
bar :: proc() -> int {
	return elsewhere()
}