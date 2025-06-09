sub tempname {
    my $lastex;
    for ^1000 {
        my $name = $*TMPDIR.child("rakudo_disassembly_" ~ (|("a".."z"),|("A".."Z"),|("0".."9")).roll(8).join("") ~ ".moarvm");
        try {
            $name.IO.open(:create, :exclusive).close();
            CATCH { default { $lastex = $!; next } }
        }
        return $name;
    }
    die "Could not come up with a free temp name file? $lastex.Str()";
}

sub MAIN($rakudoexe, $outputfile, *@extra_args) {
    my $code = @extra_args.pop;

    # If the user passes --target, we don't want to run the output through
    # the moar --dump program.
    if @extra_args && @extra_args.any.starts-with("--target=") {
        my $output = qqx/ $rakudoexe @extra_args[0] $code /;
        $outputfile.IO.spurt($output);
        return;
    }

    # Store the moar bytecode result 
    my $tempfile = tempname;
    my $moarexe = $rakudoexe.IO.sibling("moar");

    qqx/ $rakudoexe --target=mbc --output=$tempfile $code /;

    my $output = qqx[ $moarexe --dump $tempfile ];

    $outputfile.IO.spurt($output);

    LEAVE { .IO.unlink with $tempfile }
}
