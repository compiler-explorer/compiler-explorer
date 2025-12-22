# Copyright (c) 2026, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
