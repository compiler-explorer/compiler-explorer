       identification division.
       program-id. Max_array.
       data division.
       working-storage section.
           01 ws-array-1 pic s9(8) occurs 65535 times.
           01 ws-array-2 pic s9(8) occurs 65535 times.
           01 i pic s9(8) comp.
       procedure division.
           move 0 to i.
           perform varying i from 1 by 1 until i > 65535
               if ws-array-1(i) > ws-array-2(i)
                   move ws-array-1(i) to ws-array-2(i)
               end-if
           end-perform.
           stop run.
