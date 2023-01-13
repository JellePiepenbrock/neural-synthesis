#!/usr/bin/perl
while($_=<>)
{    
    chomp; 
    m/^symbols\( *([^ ,]+) *, *\[(.*)\] *, *\[(.*)\] *\)\./ 
					      or die "Bad symbols info: $_";
    ($ref, $fsyms, $psyms) = ($1, $2, $3);
    my @psyms = split(/\,/, $psyms);
    my @fsyms = split(/\,/, $fsyms);
    
    ## this now also remembers arity and symbol kind in %gsymarity
    foreach $sym (@psyms)
    {
	$sym =~ m/^ *([^\/]+)[\/]([^\/]+)[\/].*/ or die "Bad symbol $sym in $_";
	$gpreds{$1 . "__" . $2}++;
    }
    foreach $sym (@fsyms)
    {
	$sym =~ m/^ *([^\/]+)[\/]([^\/]+)[\/].*/ or die "Bad symbol $sym in $_";
	if($2 > 0)
	{ $gfuncs{$1 . "__" . $2}++; }
	else { $gconsts{$1 . "__" . $2}++; }
    }
}

print "PREDS\n";
print join("\n", (sort keys %gpreds)),"\n";
print "FUNCS\n";
print join("\n", (sort keys %gfuncs)),"\n";
print "CONSTS\n";
print join("\n", (sort keys %gconsts)),"\n";
