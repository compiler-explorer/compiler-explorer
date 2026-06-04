// Type your code here, or load an example.
// At higher optimization levels, procedures may be
// automatically inlined and will not show up in the
// output. Use the `@(export)` attribute on procedures
// you wish to see in the output.

package main

@(export)
square :: proc(num: int) -> int {
	return num * num
}
