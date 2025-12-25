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

use B qw(main_cv SVf_FAKE);
use B::Concise qw(walk_output compile);

my $filename;

# avoid modern perl features so it can work with old, old perls
# tested with 5.12, 5.18, 5.40.0, 5.42.0
sub import {
    shift; # class
    $filename = shift
	or die "diswrapper: Missing filename";
}

CHECK {
    $filename
	or die "diswrapper: Usage -Mdiswrapper,outputfilename";

    open my $fh, ">", $filename
	or die "Cannot create $filename: $!";
    binmode $fh, ":utf8";
    walk_output($fh);

    my @args;
    my $order = "basic";
    for my $arg (@ARGV) {
	if ($arg =~ /^-(basic|exec|tree)$/) {
	    $order = $1;
	    push @args, $arg;
	}
	elsif ($arg =~ /^-(?:concise|terse|linenoise|debug|
			  compact|loose|
			  base[0-9]+|bigendian|littleendian|
			  src)$/x) {
	    push @args, $arg;
	}
	else {
	    print STDERR "Unsupported argument: $arg\n";
	}
    }
    my $walker = compile('-main', @args);
    $walker->();

    # build our own list of subs to dump
    # avoid trying to dump imports
    my @subs;
    {
	my $add_lex;
	$add_lex = sub {
	    my ($prefix, $cvobj) = @_;

	    my $padlist = $cvobj->PADLIST;
	    my $padnames = $padlist->ARRAYelt(0);
	    my $padvals = $padlist->ARRAYelt(1);

	    # look for lexical subs in the PAD
	    my $index = -1;
	    for my $padname ($padnames->ARRAY) {
		++$index;
		UNIVERSAL::isa($padname, "B::SPECIAL")
		    and next; # null entry
		$padname->PV
		    or next; # not a named/typed thing
		$padname->PV =~ /^&./
		    or next; # not a lexical sub or no name
		$padname->FLAGS & SVf_FAKE
		    and next; # an outer lexical sub
		my $cvl;
		if ($padname->can("PROTOCV")) {
		    $cvl = $padname->PROTOCV
			or next;
		}
		else {
		    # before the PADNAMELIST re-work
		    $padname->can("MAGIC")
			or next;
		    $cvl = $padname->MAGIC->OBJ
			or next;
		}
		UNIVERSAL::isa($cvl, "B::CV")
		    or next;
		$cvl->START
		    or next; # not defined
		my $name = "$prefix/my sub " . substr($padname->PV, 1);
		push @subs, [ $name, $cvl ];
		$add_lex->($name, $cvl);
	    }

	    # look for anonymous subs in the PAD
	    $index = -1;
	    for my $padname ($padnames->ARRAY) {
		++$index;
		UNIVERSAL::isa($padname, "B::SPECIAL")
		    and next; # null entry
		$padname->PV
		    or next; # not typed
		$padname->PV eq "&"
		    or next; # not named or not a sub
		my $cvl = $padvals->ARRAYelt($index);
		UNIVERSAL::isa($cvl, "B::CV")
		    or next;
		my $name = "$prefix/anoncode $index";
		push @subs, [ $name, $cvl ];
		$add_lex->($name, $cvl);
	    }
	};

	no strict "refs";
	# lexical subs from the top level
	$add_lex->("main program", main_cv);
	# search the main package for subs to dump
	for my $name (keys %::) {
	    my $glob = B::svref_2object(\*{"::$name"});
	    $glob->is_empty
		and next; # no GP
	    my $cv = $glob->CV # has a CV
		or next;
	    UNIVERSAL::isa($cv, "B::CV") # which is a CV (not a constant)
		or next;
	    $cv->START # and it has a body (defined and isn't an XSUB)
		or next;
	    my $st = $cv->STASH
		or next;
	    UNIVERSAL::isa($st, "B::HV")
		or next;
	    $st->NAME eq "main" # defined in main rather than an import
		or next;
	    push @subs, [ $name, $cv ];
	    $add_lex->($name, $cv);
	}
    }

    for my $entry (@subs) {
	my ($name, $cvobj) = @$entry;
	print $fh "$name:\n";
	B::Concise::concise_cv_obj($order, $cvobj, $name);
    }

    close $fh
	or die "Cannot close $filename: $!";
}

1;
