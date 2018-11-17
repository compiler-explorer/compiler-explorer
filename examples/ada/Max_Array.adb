procedure Max_Array is 
    type Integer_Array is array(Natural range <>) of Integer;
    procedure Max_Array(x,y : in out Integer_Array) is
    begin
        for i in x'range loop
            x(i) := (if (y(i) > x(i)) then y(i) else x(i));
        end loop;
    end Max_Array;
begin
    null;
end Max_Array;