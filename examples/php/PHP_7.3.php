<?php

/**
 * New in PHP 7.3.0
 */

//Support list() reference assignments of the form:
//RFC: https://wiki.php.net/rfc/list_reference_assignment
list(&$a, list(&$b, $c)) = $d;
