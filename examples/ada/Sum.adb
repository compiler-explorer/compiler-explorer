procedure Sum is

    type Integer_Array is array(Natural range <>) of Integer;
    function Sum(input : in Integer_Array) return Natural is
        sum : Natural := 0;
    begin
        for i in input'range loop
            sum := sum + input(i);
        end loop;
        return sum;
    end Sum;
begin
    null;
end Sum;