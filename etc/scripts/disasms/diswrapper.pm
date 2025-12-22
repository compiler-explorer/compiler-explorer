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
    #local *STDOUT = *STDOUT;
    #open my $save, ">&", \*STDOUT
    #or die "diswrapper: Cannot save STDOUT: $!";
    #open STDOUT, ">", \$buf;

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
    my $walker = compile('-concise', '-main', @names);
    $walker->();

    open my $fh, ">", $filename
	or die "Cannot create $filename: $!";
    print $fh $buf;
    close $fh
	or die "Cannot close $filename: $!";
}

1;
