package main

import "core:fmt"

test_proc :: proc() {
}

@(require)
test_proc1 :: proc() {
}

@require
test_proc2 :: proc() {
}

main :: proc() {
    fmt.println("asd")
}
