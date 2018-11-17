-- Type your code here, or load an example.
function Square(num : Integer) return Integer is 
begin
    return num**2;
end Square;

-- Ada 2012 provides Expressiion Functions 
-- (http://www.ada-auth.org/standards/12rm/html/RM-6-8.html)
-- as a short hand for functions whose body consists of a 
-- single return statement.
-- function Square(num : Integer) return Integer is (num**2);
