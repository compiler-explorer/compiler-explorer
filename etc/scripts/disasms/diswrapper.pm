package diswrapper;
use strict;
use warnings;

use B qw(main_cv);
use B::Concise qw(walk_output compile);

my $filename;

# avoid modern perl features so it can work with old, old perls
sub import {
    shift; # class
    $filename = shift
	or die "diswrapper: Missing filename";
}

CHECK {
    $filename
	or die "diswrapper: Usage -Mdiswrapper,outputfilename";
    my $buf;

    walk_output(\$buf);
    # build our own list of subs to dump
    # avoid trying to dump imports
    my @names;
    {
	no strict "refs";
	for my $name (keys %::) {
	    my $glob = B::svref_2object(\*{"::$name"});
	    if (!$glob->is_empty
		&& (my $cv = $glob->CV)) {
		if (UNIVERSAL::isa($cv, "B::CV")
		    && $cv->START  # not an XSUB and defined
		    && (my $st = $cv->STASH)) {  # has a stack
		    if (UNIVERSAL::isa($st, "B::HV") # XSUBs don't
			&& $st->NAME eq "main") {
			push @names, "main::$name";
		    }
		}
	    }
	}
    }
    my @args;
    for my $arg (@ARGV) {
	if ($arg =~ /^-(?:concise|terse|linenoise|debug|
			  basic|exec|tree|
			  compact|loose|
			  base[0-9]+|bigendian|littleendian|
			  src)$/x) {
	    push @args, $arg;
	}
	else {
	    print STDERR "Unsupported argument: $arg\n";
	}
    }
    my $walker = compile('-main', @args, @names);
    $walker->();

    open my $fh, ">", $filename
	or die "Cannot create $filename: $!";
    print $fh $buf;
    close $fh
	or die "Cannot close $filename: $!";
}

1;
