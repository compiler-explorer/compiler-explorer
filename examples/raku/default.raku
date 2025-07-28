
sub square($number) {
    $number * $number
}

multi sub MAIN(Int $foo) {
    say "When no one was looking, Lex Luthor took $foo cakes.";
    say "The square of that is &square($foo) cakes.";
    say "It's also roughly as much as $($foo div 10) tens.";
    say "And that's terrible.";
}

multi sub MAIN() {
    say "You can pass a number on the commandline if you like!"
}
