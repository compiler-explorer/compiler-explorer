Finding entry points
Branch analysis from position: 0
1 jumps found. (Code = 62) Position 1 = -2
filename:       examples/php/Namespaces_and_traits.php
function name:  (null)
number of ops:  18
compiled vars:  !0 = $a
line     #* E I O op                           fetch          ext  return  operands
-------------------------------------------------------------------------------------
   6     0  E >   NOP
  12     1        DECLARE_CLASS                                    $2      'ns%5Ca'
  13     2        ADD_TRAIT                                                $2, 'NS%5CcanAdd'
  12     3        BIND_TRAITS                                              $2
  21     4        INIT_STATIC_METHOD_CALL                                  'NS%5CA', 'add_two_numbers'
         5        SEND_VAL_EX                                              1
         6        SEND_VAL_EX                                              2
         7        DO_FCALL                                      0  $3
         8        ECHO                                                     $3
  22     9        NEW                                              $4      :5
        10        DO_FCALL                                      0
        11        ASSIGN                                                   !0, $4
  23    12        INIT_METHOD_CALL                                         !0, 'multiple_two_numbers'
        13        SEND_VAL_EX                                              1
        14        SEND_VAL_EX                                              2
        15        DO_FCALL                                      0  $7
        16        ECHO                                                     $7
  25    17      > RETURN                                                   1

branch: #  0; line:     6-   25; sop:     0; eop:    17; out0:  -2
path #1: 0,
Class NS\canAdd:
Function add_two_numbers:
Finding entry points
Branch analysis from position: 0
1 jumps found. (Code = 62) Position 1 = -2
filename:       examples/php/Namespaces_and_traits.php
function name:  add_two_numbers
number of ops:  7
compiled vars:  !0 = $num1, !1 = $num2
line     #* E I O op                           fetch          ext  return  operands
-------------------------------------------------------------------------------------
   7     0  E >   RECV                                             !0
         1        RECV                                             !1
   8     2        ADD                                              ~2      !0, !1
         3        VERIFY_RETURN_TYPE                                       ~2
         4      > RETURN                                                   ~2
   9     5*       VERIFY_RETURN_TYPE
         6*     > RETURN                                                   null

branch: #  0; line:     7-    9; sop:     0; eop:     6
path #1: 0,
End of function add_two_numbers

End of class NS\canAdd.

Class NS\A:
Function multiple_two_numbers:
Finding entry points
Branch analysis from position: 0
1 jumps found. (Code = 62) Position 1 = -2
filename:       examples/php/Namespaces_and_traits.php
function name:  multiple_two_numbers
number of ops:  7
compiled vars:  !0 = $a, !1 = $b
line     #* E I O op                           fetch          ext  return  operands
-------------------------------------------------------------------------------------
  14     0  E >   RECV                                             !0
         1        RECV                                             !1
  15     2        MUL                                              ~2      !0, !1
         3        VERIFY_RETURN_TYPE                                       ~2
         4      > RETURN                                                   ~2
  16     5*       VERIFY_RETURN_TYPE
         6*     > RETURN                                                   null

branch: #  0; line:    14-   16; sop:     0; eop:     6
path #1: 0,
End of function multiple_two_numbers

End of class NS\A.
